"""
"""
import pickle

import intake
from common import remove_outliers
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 20
cat = intake.open_catalog("../leaf-level.yaml")
src = cat["Lin2015"]
if not src.is_persisted:
    src.persist()
df = src.read()
df = df[["CO2S", "PARin", "VPD", "Cond"]]
df = remove_outliers(df, verbose=True)
x = df.drop(columns="Cond")
y = df["Cond"]
regr = RandomForestRegressor(random_state=RANDOM_STATE, bootstrap=True)
regr.fit(x, y)
with open("random_forest_regressor.pkl", "wb") as pkl:
    pickle.dump(regr, pkl)
