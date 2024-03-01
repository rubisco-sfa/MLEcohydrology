import os
import random
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras_tuner.tuners import GridSearch
from tensorflow.keras import layers

# Read in data
os.system("sed -i 's/canopy //g' canopy.txt")
df = pd.read_csv(
    "canopy.txt",
    header=0,
    names=[
        "t_veg0",
        "forc_lwrad",
        "forc_t",
        "forc_u",
        "forc_v",
        "forc_pco2",
        "smpso",
        "smpsc",
        "sabv",
        "t_grnd",
        "iter",
        "t_veg",
    ],
    on_bad_lines="skip",
)
df = df.groupby([pd.cut(df[col], 30) for col in df.columns], observed=True).mean()

if not Path("canopy_pairplot.png").is_file():
    sns.pairplot(df.select_dtypes(include=np.number), diag_kind="hist")
    plt.savefig("canopy_pairplot.png")

# Setup for neural net
df = df.drop(columns="forc_v")  # equal to forc_u
df = df.drop(columns=["smpso", "smpsc"])  # constant
df = df.drop(columns="iter")  # outputs
COLUMNS = list(df.columns)
TARGET = "t_veg"
dfs = df[COLUMNS]

# layers are initialized randomly, to ensure reproducible results we need to set
# a lot of seeds
RANDOM_STATE = 1
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# split test and train data
train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
test = dfs.drop(train.index)
train_features = train.copy()
train_labels = train_features.pop(TARGET)
test_features = test.copy()
test_labels = test_features.pop(TARGET)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_features)

msd = ((df["t_veg"] - df["t_veg0"]) ** 2).mean()


def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) / msd


# routine that builds a neural net
def build_model(hp, normalizer):
    nlayers = hp.Int("nlayers", min_value=1, max_value=2)
    units = hp.Int(
        "neurons_per_layer", min_value=2, max_value=1024, step=2, sampling="log"
    )
    activation = hp.Choice("activation", ["sigmoid"])
    model = [normalizer]
    model += [layers.Dense(units=units, activation=activation) for _ in range(nlayers)]
    model += [layers.Dense(1, activation="linear")]
    model = tf.keras.Sequential(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3])),
        loss=custom_loss,
    )
    return model


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


tuner = GridSearch(
    partial(build_model, normalizer=normalizer),
    objective="val_loss",
    max_trials=20,
    executions_per_trial=1,
    max_model_size=0.1 * len(train_labels),
    max_consecutive_failed_trials=100,
    directory="_project",
    project_name="canopy_surrogate",
)
tuner.search(
    train_features,
    train_labels,
    epochs=100,
    validation_split=0.2,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    ],
)

tuner.results_summary()

for i, m in enumerate(tuner.get_best_models(7)):

    test_predictions = m.predict(test_features)
    train_predictions = m.predict(train_features)
    t_veg = m.predict(df[[c for c in COLUMNS if c != TARGET]])

    mse = m.evaluate(test_features, test_labels)
    corr = np.corrcoef(test_labels, test_predictions.flatten())[0, 1]

    fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    ax[0, 0].scatter(train_labels, train_predictions, label="train", s=7)
    ax[0, 0].scatter(test_labels, test_predictions, label="test", s=5)
    vmax = max(test_labels.max(), test_predictions.max())
    vmin = min(test_labels.min(), test_predictions.min())
    ax[0, 0].plot([vmin, vmax], [vmin, vmax], "--k")
    ax[0, 0].legend(loc=4)
    ax[0, 0].set_xlabel("True Values")
    ax[0, 0].set_ylabel("Predicted Values")
    label = f"""test_MSE={mse:1.2e}
    R={corr:.3f}"""
    ax[0, 0].text(
        0.02,
        0.98,
        label,
        ha="left",
        va="top",
        transform=ax[0, 0].transAxes,
    )

    ax[0, 1].axis("off")
    ax[1, 0].scatter(df["t_veg0"], df["t_veg"], s=5)
    ax[1, 1].scatter(df["t_veg0"], t_veg, s=5)
    ax[1, 0].set_xlabel("t_veg0")
    ax[1, 1].set_xlabel("t_veg0")
    ax[1, 0].set_ylabel("t_veg")
    ax[1, 1].set_ylabel("t_veg predicted")
    fig.suptitle(f"Parameters: {m.count_params()}")
    fig.savefig(f"perf{i:02d}.png")
    plt.close()
