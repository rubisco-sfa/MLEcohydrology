import os
import random
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras import layers

if not Path("photo.parquet").is_file():
    os.system("wget wget https://www.climatemodeling.org/~nate/photo.csv.gz")
    os.system("gunzip photo.csv.gz")
    os.system(r"sed -i -e 's/hybrid  //g' photo.csv")
    df = pd.read_csv(
        "photo.csv",
        names=[
            "x0",  # initial guess of the solution
            "lmr_z",  # canopy layer: leaf maintenance respiration rate (umol CO2/m2/s)
            "par_z",  # par absorbed per unit lai for canopy layer (w/m2)
            "rh_can",  # canopy air relative humidity
            "gb_mol",  # leaf boundary layer conductance (umol H2O/m2/s)
            "je",  # electron transport rate (umol electrons/m2/s)
            "cair",  # atmospheric CO2 partial pressure (Pa)
            "oair",  # atmospheric O2 partial pressure (Pa)
            "p",  # pft
            "iv",  # c3/c4
            "c",  # column
            "gs_mol",  # leaf stomatal conductance (umol H2O/m2/s)
            "iter",  # number of iterations used, for record only
            "xf",  # final value of the solution
        ],
    ).dropna()
    df.to_parquet("photo.parquet")
    os.system("rm -f photo.csv.gz photo.csv")

df = pd.read_parquet("photo.parquet")
if not Path("elm.png").is_file():
    sns.pairplot(df.select_dtypes(include=np.number), diag_kind="hist")
    plt.savefig("elm.png")
    plt.close()

# get ready for NNs
df = df.drop(columns=["x0"])  # we are trying to do better than this
df = df.drop(columns=["iter", "gs_mol"])  # these are outputs
df = df.drop(columns=["p", "iv", "c"])  # these are all constant
df = df.drop(columns=["oair"])  # perfectly linearly related to cair
df = df.iloc[::10]
COLUMNS = list(df.columns)
RANDOM_STATE = 1
TARGET = "xf"
dfs = df[COLUMNS]

# layers are initialized randomly, to ensure reproducible results we need to set
# a lot of seeds
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


def build_model(hp, normalizer):
    nlayers = hp.Int("nlayers", min_value=1, max_value=4)
    units = hp.Int(
        "neurons_per_layer", min_value=8, max_value=128, step=2, sampling="log"
    )
    activation = hp.Choice("activation", ["relu", "sigmoid"])
    model = [normalizer]
    for i in range(nlayers):
        model.append(layers.Dense(units=units, activation=activation))
    model.append(layers.Dense(1, activation="linear"))
    model = tf.keras.Sequential(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
    )
    return model


tuner = BayesianOptimization(
    partial(build_model, normalizer=normalizer),
    objective="val_loss",
    max_trials=20,
    executions_per_trial=2,
    max_model_size=0.1 * len(train_labels),
    max_consecutive_failed_trials=20,
    directory="_project",
    project_name="photosynthesis_surrogate",
)
tuner.search(
    train_features,
    train_labels,
    epochs=10,
    validation_split=0.2,
    verbose=2,
)
tuner.results_summary()
