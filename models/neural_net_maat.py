"""."""
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 20
plt.rcParams.update({"font.size": 16})


def linear_model(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_features: pd.DataFrame,
):
    """."""
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_features)
    model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="mean_absolute_error",
    )
    model.fit(
        train_features,
        train_labels,
        epochs=100,
        verbose=1,
        validation_split=0.2,
    )
    return model.predict(test_features).flatten()


def dnn_model(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_features: pd.DataFrame,
):
    """."""
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_features)
    model = tf.keras.Sequential(
        [
            normalizer,
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="mean_absolute_error",
    )
    model.fit(
        train_features,
        train_labels,
        epochs=1000,
        verbose=1,
        validation_split=0.2,
    )
    return model.predict(test_features).flatten()


# read in the leaf level data
cat = intake.open_catalog("../leaf-level.yaml")
src = cat["Lin2015"]
if not src.is_persisted:
    src.persist()
df = src.read().dropna()


if True:
    dfs = df[["VPD", "CO2S", "Tleaf", "Photo", "Cond"]]
    train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
    test = dfs.drop(train.index)
    train_features = train.copy()
    train_labels = train_features.pop("Cond")
    test_features = test.copy()
    test_labels = test_features.pop("Cond")
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    prediction = linear_model(train_features, train_labels, test_features)
    ax.scatter(test_labels, prediction)
    ax.plot([0, test_labels.max()], [0, test_labels.max()], "--k")
    ax.axis("equal")
    ax.set_xlabel("Lin2015 Cond")
    ax.set_ylabel("Linear Model Cond")
    ax.set_title(f"$R^{2}$ = {np.corrcoef(prediction,test_labels)[0,1]:.3f}")
    fig.savefig("nn_linear.png")
    plt.close()

if True:
    dfs = df[["VPD", "CO2S", "Tleaf", "Photo", "Species", "Cond"]]
    dfs = pd.get_dummies(dfs, columns=["Species"])
    train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
    test = dfs.drop(train.index)
    train_features = train.copy()
    train_labels = train_features.pop("Cond")
    test_features = test.copy()
    test_labels = test_features.pop("Cond")
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    prediction = linear_model(train_features, train_labels, test_features)
    ax.scatter(test_labels, prediction)
    ax.plot([0, test_labels.max()], [0, test_labels.max()], "--k")
    ax.axis("equal")
    ax.set_xlabel("Lin2015 Cond")
    ax.set_ylabel("Species Linear Model Cond")
    ax.set_title(f"$R^{2}$ = {np.corrcoef(prediction,test_labels)[0,1]:.3f}")
    fig.savefig("nn_linear_species.png")
    plt.close()

if True:
    dfs = df[["VPD", "CO2S", "Tleaf", "Photo", "Cond"]]
    train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
    test = dfs.drop(train.index)
    train_features = train.copy()
    train_labels = train_features.pop("Cond")
    test_features = test.copy()
    test_labels = test_features.pop("Cond")
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    prediction = dnn_model(train_features, train_labels, test_features)
    ax.scatter(test_labels, prediction)
    ax.plot([0, test_labels.max()], [0, test_labels.max()], "--k")
    ax.axis("equal")
    ax.set_xlabel("Lin2015 Cond")
    ax.set_ylabel("DNN Model Cond")
    ax.set_title(f"$R^{2}$ = {np.corrcoef(prediction,test_labels)[0,1]:.3f}")
    fig.savefig("dnn.png")
    plt.close()

if True:
    dfs = df[["VPD", "CO2S", "Tleaf", "Photo", "Species", "Cond"]]
    dfs = pd.get_dummies(dfs, columns=["Species"])
    train = dfs.sample(frac=0.8, random_state=RANDOM_STATE)
    test = dfs.drop(train.index)
    train_features = train.copy()
    train_labels = train_features.pop("Cond")
    test_features = test.copy()
    test_labels = test_features.pop("Cond")
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    prediction = dnn_model(train_features, train_labels, test_features)
    ax.scatter(test_labels, prediction)
    ax.plot([0, test_labels.max()], [0, test_labels.max()], "--k")
    ax.axis("equal")
    ax.set_xlabel("Lin2015 Cond")
    ax.set_ylabel("Species DNN Model Cond")
    ax.set_title(f"$R^{2}$ = {np.corrcoef(prediction,test_labels)[0,1]:.3f}")
    fig.savefig("dnn_species.png")
    plt.close()
