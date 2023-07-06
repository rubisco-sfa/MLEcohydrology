import itertools
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from common import get_number_of_parameters, remove_outliers
from tensorflow.keras import layers


def round_sigdigits(x, digits=1):
    """Return x rounded to the specified number of significant digits."""
    return round(x, -int(np.floor(np.log10(np.abs(x)) - digits + 1)))


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min.

    Parameters
    ----------
    rate_patience: int
        The number of epochs to wait after a minimum before reducing the
        learning rate.
    end_patience: int
        The number of epochs to wait after a minimum before ending the training.
    max_reducation: float
        The maximum ratio the initial learning rate may be reduced.

    """

    def __init__(self, rate_patience=0, end_patience=0, max_reduction=1e-2):
        super().__init__()
        self.rate_patience = rate_patience
        self.end_patience = end_patience
        self.max_reduction = max_reduction
        self.tr0 = None
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.tr0 = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs=None):
        if self.wait >= self.rate_patience:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            scheduled_lr = max(0.5 * lr, self.tr0 * self.max_reduction)
            if np.isclose(lr, scheduled_lr):
                return
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print(f"Learning rate reduction {lr:1.3e} --> {scheduled_lr:1.3e}")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.end_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch+1}: early stopping")


COLUMNS = ["VPD", "CO2S", "Tleaf", "PARin"]
RANDOM_STATE = 1
TARGET = "Photo"
if len(sys.argv) > 1:
    TARGET = sys.argv[1]

# layers are initialized randomly, to ensure reproducible results we need to set
# a lot of seeds
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# read in the leaf level data
df = pd.read_csv("../maat/Lin2015_cleaned_BDTT.csv")
COLUMNS.append(TARGET)
df = remove_outliers(df, columns=COLUMNS, verbose=True)
dfs = df[COLUMNS]

# split test and train data
train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
test = dfs.drop(train.index)
train_features = train.copy()
train_labels = train_features.pop(TARGET)
test_features = test.copy()
test_labels = test_features.pop(TARGET)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_features)

# we are going to train a few nets, keep some stats in a dataframe so we can pick the
# best at the end of the process.
results = []

#  setup Latin hypercube parameter experiment
learning_rates = [1e-2]
neurons = 2.0 ** np.arange(3, 6)
nlayers = [1, 2]
activations = ["relu", "sigmoid"]
count = 0
for learning_rate, neuron, nlayer, activation in itertools.product(
    learning_rates, neurons, nlayers, activations
):
    # build the model
    model = [normalizer]
    model += [layers.Dense(neuron, activation=activation) for _ in range(nlayer)]
    model += [layers.Dense(1)]
    model = tf.keras.Sequential(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    # to avoid overfitting, only accept models where the number of parameters is
    # less than 10% of the size of the training data
    num_params = get_number_of_parameters(model)
    if num_params > 0.1 * len(train_labels):
        continue
    model.fit(
        train_features,
        train_labels,
        epochs=1000,
        validation_split=0.1,
        verbose=2,
        callbacks=[
            EarlyStoppingAtMinLoss(rate_patience=3, end_patience=5, max_reduction=1e-3)
        ],
    )
    history = model.history.history  # this will get reset when we evaluate, so store
    test_predictions = model.predict(test_features)
    train_predictions = model.predict(train_features)
    mse = model.evaluate(test_features, test_labels)
    corr = np.corrcoef(test_labels, test_predictions.flatten())[0, 1]
    df[f"{TARGET}{count:02d}"] = model.predict(dfs.drop(columns=TARGET))

    # saving the error measures to 3 sigdigits so we can sort out models which provide
    # approximately the same error.
    results.append(
        {
            "name": f"{TARGET}{count:02d}",
            "num_parameters": num_params,
            "mean_squared_error": round_sigdigits(mse, 3),
            "correlation": round_sigdigits(corr, 3),
            "activation": activation,
            "model": model,
        }
    )

    # plot the training history
    fig, axs = plt.subplots(figsize=(10, 5), ncols=2, tight_layout=True)
    last_epoch = None
    for loss, history in history.items():
        axs[0].semilogy(history, label=loss)
        if last_epoch is None:
            last_epoch = len(history)
    label = f"""$\\alpha_0=${learning_rate:1.1e}
    N={num_params} ({nlayer})
    f={num_params/len(train_labels)*100:.1f}%
    {activation}
    """
    axs[0].plot(last_epoch - 1, mse, "ok", label="test_loss", ms=3)
    axs[0].legend(loc=2)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Mean Square Error")
    axs[0].text(
        0.98,
        0.98,
        label,
        ha="right",
        va="top",
        transform=axs[0].transAxes,
    )
    axs[0].set_title("Training Information")

    # plot the performance of the train and test data
    axs[1].scatter(train_labels, train_predictions, label="train", s=7)
    axs[1].scatter(test_labels, test_predictions, label="test", s=5)
    vmax = max(test_labels.max(), test_predictions.max())
    axs[1].plot([0, vmax], [0, vmax], "--k")
    axs[1].legend(loc=4)
    axs[1].set_xlabel("True Values")
    axs[1].set_ylabel("Predicted Values")
    label = f"""test_MSE={mse:1.2e}
R={corr:.3f}"""
    axs[1].text(
        0.02,
        0.98,
        label,
        ha="left",
        va="top",
        transform=axs[1].transAxes,
    )
    axs[1].set_title("Performance")
    fig.suptitle(TARGET)
    fig.savefig(f"{TARGET}{count:03d}.png")
    plt.close()
    count += 1

# save the 'best' model
results = pd.DataFrame(results)
print(
    results.sort_values(["mean_squared_error", "correlation", "num_parameters"]).iloc[0]
)
df.to_csv("Lin2015_BDTT_Photo_NN.csv")
