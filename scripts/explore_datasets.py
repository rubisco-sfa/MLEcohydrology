"""."""

import pandas as pd
import plotly
import plotly.express as px

df = pd.read_parquet("Lin2015_clm5.parquet")
n0 = len(df)
df = df[df[["Photo", "Tleaf", "PARin"]].notna().all(axis=1)]
df = df[df["PARin"] > 0]
print(len(df))

fig = px.scatter(
    df,
    title="Lin2015 Leaf Level Flux Data",
    x="Tleaf",
    y="Photo",
    color="Species",
    size="PARin",
    hover_name="Species",
    hover_data={
        "Species": False,
        "PFT": False,
        "Tair": ":.1f",
        "Tleaf": ":.1f",
        "VPD": ":.2f",
        "PARin": ":.0f",
        "CO2S": ":.0f",
    },
    facet_col="PFT",
    facet_col_wrap=5,
)
fig.update_layout(
    font_size=24,
    hoverlabel=dict(
        font_size=20,
    ),
)
plotly.offline.plot(fig, filename="Lin2015_pft.html")
