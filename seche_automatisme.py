import pandas as pd
import sklearn.ensemble as rf
import sklearn.model_selection as ms
import numpy as np
import sklearn.preprocessing as pp
import pickle
import matplotlib.pyplot as plt

from datetime import datetime

dataframe = pd.read_excel("data/seche/Automatisme2024-11-201740.xlsx", skiprows=[1], na_values="NS")
dataframe.rename(columns={'Unnamed: 0': 't'}, inplace=True )
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
dataframe["t"] = pd.to_datetime(dataframe["t"], format='%d/%m/%Y %H:%M:%S')
dataframe["t"] = dataframe["t"].dt.second / 600

print(dataframe.head(5))
print(dataframe.describe())

# dataframe = dataframe.drop("UTI.METEO.HYGROMETRE.MESURE", axis=1)

drop_list = ["RCU.SS_UPE.PT0341025.MESURE",
"RCU.SS_UPE.PT0341135.MESURE",
"RCU.SS_UPE.TT0341024.MESURE",
"RCU.SS_UPE.FT0341014.MESURE",
"RCU.SS_UPE.LT0341008.MESURE",
"RCU.SS_UPE.LT0341035.MESURE",
"RCU.SS_UPE.TT0341120.MESURE",
"RCU.SS_UPE.TT0341058.MESURE",
"RCU.SS_UPE.TT0341066.MESURE",
"RCU.SS_UPE.FT0341103.MESURE",
"RCU.SS_UPE.FT0341107.MESURE",
"RCU.SS_UPE.FQI0341115.TE_aller",
"RCU.SS_UPE.FQI0341115.TE_retour",
"RCU.SS_UPE.PT0341077.MESURE",
"RCU.SS_UPE.PT0341084.MESURE",
"RCU.SS_UPE.PT0341091.MESURE",
"RCU.SS_UPE.GMP.A_press_GMP",
"RCU.SS_UPE.P0341076.Courant",
"RCU.SS_UPE.P0341083.Courant",
"RCU.SS_UPE.P0341090.Courant",
"RCU.SS_UPE.TT0341038.MESURE",
"RCU.SS_UPE.TT0341011.MESURE",
"RCU.SS_UPE.FT0341041.MESURE",
"RCU.SS_UPE.TT0341101.MESURE",
"RCU.SS_UPE.FQI0341115.Debit",
"RCU.SS_FERIE.FQI0341219.TE_aller",
"RCU.SS_FERIE.FQI0341219.TE_retour",
"RCU.SS_FERIE.FQI0341219.Debit",
"UTI.METEO.BAROMETRE.MESURE",
"UTI.METEO.ANEMOMETRE.MESURE",
"UTI.METEO.GIROUETTE.MESURE",
"UTI.METEO.HYGROMETRE.MESURE"]

dataframe = dataframe.drop(drop_list, axis=1)

# dataframe = dataframe.drop("t", axis=1)
dataframe = dataframe.dropna()
print(dataframe.isna().sum())
dataframe.to_csv("data/seche/cleaned.csv")

name = "RCU.SS_UPE.FQI0341115.Puissance"
y = dataframe[name]
x = dataframe.drop(name, axis=1)

scaler = pp.RobustScaler()
scaler.fit(x)
xscaler = scaler.transform(x)

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(xscaler, y, train_size=0.8, test_size=0.2)

model = rf.RandomForestRegressor()
model.fit(xtrain, ytrain)
train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)
print(train_score, test_score)

with open("data/seche/rf.pickle", "wb") as f:
    pickle.dump(model, f)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()












