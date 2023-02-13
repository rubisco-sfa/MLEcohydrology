"""Import the Saunders2021 data and prepare it for use in MAAT."""
import os

import intake

if not os.path.isdir("./logs"):
    os.makedirs("logs")
if not os.path.isdir("./results"):
    os.makedirs("results")

cat = intake.open_catalog("../../leaf-level.yaml")
df = cat["Saunders2021"].read()
df["PARin"] = df["solar"] * 0.45 * 4.57
df = df[["PARin", "Tleaf", "VPDleaf"]]
df.to_csv("Saunders2021_cleaned.csv", index=False)
