import pandas as pd
import sklearn.ensemble as rf
import sklearn.model_selection as ms
import numpy as np
import sklearn.preprocessing as pp
import pickle
import matplotlib.pyplot as plt

dataframe = pd.read_excel("data/seche/Automatisme2024-11-201740.xlsx", skiprows=[1], na_values="NS")  #, parse_dates=True, date_format='%Y/%m/%d %H:%M:%S')
dataframe.rename( columns={'Unnamed: 0': 't'}, inplace=True )
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
dataframe.set_index("t")
print(dataframe.head(5))
print(dataframe.describe())

dataframe = dataframe.drop("UTI.METEO.HYGROMETRE.MESURE", axis=1)
dataframe = dataframe.drop("t", axis=1)
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
plt.xticks(rotation=45)
plt.show()












