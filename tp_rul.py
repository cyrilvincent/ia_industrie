import pandas as pd
import sklearn.preprocessing as pp
import tensorflow as tf

dataframe = pd.read_csv("data/rul/Battery_RUL.csv", index_col="Cycle_Index")
y = dataframe.RUL
x = dataframe.drop("RUL", axis=1)

scaler = pp.StandardScaler()
scaler.fit(x)
xscaler = scaler.transform(x)

# Créer le modèle
# Le compiler
# Fit
# Score

