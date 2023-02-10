"""Machine learning models perform better than traditional empirical models for
stomatal conductance when applied to multiple tree species across different
forest biomes, 'Trees, Forests and People', 6 (2021).

Line references refer to Saunders code listed here:

https://github.com/altazietsman/ML-stomatal-conductance-models/blob/master/Model%20development/preprocessing.py

"""
import time

import intake
import pandas as pd
from common import remove_outliers
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 20

cat = intake.open_catalog("../leaf-level.yaml")

### Saunders2021 data processing

src = cat["Saunders2021"]
if not src.is_persisted:
    src.persist()
dfs = src.read()
dfs["PARin"] = dfs["solar"] * 0.45 * 4.57  # L26
dfs = dfs.rename(columns={"VPDleaf": "VPD"})
dfs = dfs[["PARin", "SWC", "VPD", "Cond", "Species"]]

### Anderegg2018 data processing

src = cat["Anderegg2018"]
if not src.is_persisted:
    src.persist()
dfa = src.read()
N = 1.56
M = 1 - 1 / N
ALPHA = 0.036
dfa["SWC"] = dfa["SWC"].fillna(
    1 / ((1 + (-1 * (dfa["LWPpredawn"]) / ALPHA) ** N) ** M)
)  # L17
dfa = dfa[["PARin", "SWC", "VPD", "Cond", "Species"]]

# Combine datasets and creates binary columns from unique values in Species
df = pd.concat([dfa, dfs])
df = remove_outliers(df, ["PARin", "VPD", "SWC", "Cond"], verbose=True)
df = pd.get_dummies(df, columns=["Species"])

# Their paper discusses only training with species with > 100 data entries. I
# found no evidence of this in the code.
x = df.drop(columns="Cond")
y = df["Cond"]
x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.20, random_state=RANDOM_STATE
)

# Initialize the regressors used in the paper
regressors = {
    "MLR": LinearRegression(),
    "DT": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RF": RandomForestRegressor(random_state=RANDOM_STATE, bootstrap=True),
}
regressors["RF - Bagging"] = BaggingRegressor(base_estimator=regressors["RF"])
regressors["RF - Boosting"] = AdaBoostRegressor(base_estimator=regressors["RF"])

# Loop over regressors, train and fit and record performance information
perf = []
for label, regr in regressors.items():
    tfit = time.time()
    regr.fit(x_train, y_train)
    tfit = time.time() - tfit
    y_predict = regr.predict(x_test)
    perf.append(
        {
            "method": label,
            "time": tfit,
            "MSE": mean_squared_error(y_test, y_predict),
            "R2": r2_score(y_test, y_predict),
        }
    )
perf = pd.DataFrame(perf)
print(perf)
