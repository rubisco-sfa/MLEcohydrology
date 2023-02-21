"""
"""
import pickle

import intake
import matplotlib.pyplot as plt
import numpy as np
from common import remove_outliers
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 20
cat = intake.open_catalog("../leaf-level.yaml")
src = cat["Lin2015"]
if not src.is_persisted:
    src.persist()
df = src.read()
df = df[["CO2S", "PARin", "VPD", "Photo", "Cond"]]
df = remove_outliers(df, verbose=True)
x = df.drop(columns="Cond")
y = df["Cond"]
regr = RandomForestRegressor(random_state=RANDOM_STATE, bootstrap=True)
regr.fit(x, y)
with open("random_forest_regressor.pkl", "wb") as pkl:
    pickle.dump(regr, pkl)

fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
prediction = regr.predict(x)
ax.scatter(y, prediction)
ax.plot([0, y.max()], [0, y.max()], "--k")
ax.axis("equal")
ax.set_xlabel("Lin2015 Cond")
ax.set_ylabel("Random Forest Cond")
ax.set_title(f"$R^{2}$ = {np.corrcoef(prediction,y)[0,1]:.3f}")
fig.savefig("cond_rf.png")
