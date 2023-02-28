"""."""
from difflib import get_close_matches

import intake
import numpy as np
import pandas as pd


def clean_species(dfs: pd.DataFrame):
    """Cleanup the species column to make comparisons simpler"""
    dfs["Species"] = dfs["Species"].str.replace("_", " ").str.lower().str.strip()
    return dfs


def pick_pft(row):
    """."""
    row["Funtype"] = str(row["Funtype"])
    # Assuming plantform 'savanna' can map to 'tree'
    row["Plantform"] = row["Plantform"].replace("savanna", "tree")
    pft = (
        "needleleaf "
        if (
            ("conifer" in row["Funtype"])
            or ("needle" in row["Funtype"])
            or ("pinus" in row["Species"])
        )
        else "broadleaf "
    )
    pft += row["Leafspan"] + " " + row["Plantform"] + " " + row["Tregion"].lower()
    # Handle grasses
    if "grass" in row["Plantform"]:
        pft = row["Pathway"].lower()
        pft += (" " + row["Tregion"].lower()) if row["Tregion"] == "Arctic" else ""
        pft += " grass"
    # Handle crops
    if "crop" in row["Plantform"]:
        pft = "c3 un" if row["Pathway"] == "C3" else ""
        pft += "managed rainfed crop"
    # No such pft, switch manually
    if pft == "broadleaf deciduous shrub arctic":
        pft = "broadleaf deciduous shrub boreal"
    return pft


def pick_ivt(row):
    """."""
    try:
        return df_pft[df_pft["PFT"] == row["PFT"]].index[0]
    except IndexError:
        return np.nan


def pick_abbr(row):
    """."""
    return df_pft["Acronym"].iloc[row["IVT"]]


def pick_doaa(row, allow_close_matches=False):
    """."""
    dfs = df_doaa.loc[df_doaa["Species"] == row["Species"]]

    # if we do not find what we are looking for...
    if len(dfs) == 0:
        # ...maybe it just starts with the name
        species = [
            s for s in df_doaa["Species"].unique() if s.startswith(row["Species"])
        ]
        # ...if still not, lets pick the closest match
        if allow_close_matches:
            if len(species) == 0:
                species = get_close_matches(row["Species"], df_doaa["Species"].unique())
                if len(species) > 1:
                    species = species[:1]
        if len(species) == 0:
            return "none"
        if len(species) == 1:
            dfs = df_doaa.loc[df_doaa["Species"] == species[0]]

    # just return if we have an exact match
    if len(dfs) >= 1:
        return dfs["Corresponding PFT in CLM4.5"].iloc[0]

    return "none"


cat = intake.open_catalog("../leaf-level.yaml")
df_pft = pd.read_html(
    "https://escomp.github.io/ctsm-docs/versions/release-clm5.0/html/tech_note/Ecosystem/CLM50_Tech_Note_Ecosystem.html"
)[0]
df_pft["PFT"] = (
    df_pft["Plant functional type"]
    .str.replace("-", "–")
    .str.replace(" – ", " ")
    .str.replace("1", "")
    .str.replace("2", "")
    .str.lower()
)
df_pft["Acronym"] = df_pft["Acronym"].fillna(df_pft["Plant functional type"])

df = cat["Lin2015"].read()
df = clean_species(df)

df_doaa = intake.open_excel("Species_PFT_Asignment.xlsx").read()
df_doaa = clean_species(df_doaa)

df["PFT"] = df.apply(pick_pft, axis=1)
df["IVT"] = df.apply(pick_ivt, axis=1)
df["PFT Abbreviation"] = df.apply(pick_abbr, axis=1)
df["Doaa"] = df.apply(pick_doaa, axis=1)

assert len(df[df["IVT"].isna()]["PFT"].unique()) == 0
diff = df[df["PFT Abbreviation"] != df["Doaa"]]
for g, grp in diff.groupby(["PFT Abbreviation", "Doaa"]):
    print(f"Our PFT ->  {g[0]} | {g[1]}  <- Doaa's PFT")
    print("Species involved: ", ", ".join(grp["Species"].unique()))
    print("")
