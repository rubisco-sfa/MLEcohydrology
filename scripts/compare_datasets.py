"""This script uses the intake catalog to loop through and compute pairplot
diagrams for all the numerical columns of the datasets found."""
import os

import intake
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cat = intake.open_catalog(
    "https://raw.githubusercontent.com/rubisco-sfa/MLEcohydrology/main/leaf-level.yaml"
)

for key in cat:
    if os.path.isfile(f"{key}.png"):
        print(f"Image already exists for {key}")
        continue
    print(f"Processing {key}...")
    src = cat[key]
    if not src.is_persisted:
        print("  Persisting data...")
        src.persist()
    df = src.read()
    for col in df.columns:
        if df[col].isna().all():
            df.pop(col)
    if "latitude" in df.columns:
        df = df.drop(columns=["latitude"])
    if "longitude" in df.columns:
        df = df.drop(columns=["longitude"])
    sns.pairplot(
        df.select_dtypes(include=np.number),
        diag_kind="kde",
    )
    plt.savefig(f"{key}.png")
    plt.close()
