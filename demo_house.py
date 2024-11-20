import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe.loyer / dataframe["surface"]
print(dataframe.describe())
dataframe.to_json("data/house/house.json")

# 1 data
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# 2 = créer le modèle
# model = lm.LinearRegression()
degre = 1
model = pipe.make_pipeline(pp.PolynomialFeatures(degre), lm.Ridge())

# 3 = Fit = Apprentissage
# Un seul interdit, ne JAMAIS s'entrainer sur le xtest
model.fit(xtrain, ytrain)


xsample = np.arange(400).reshape(-1, 1)
# 4 Prédiction
predicted = model.predict(xsample)

score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)

# print(model.coef_, model.intercept_)

# 5 Display
plt.scatter(dataframe["surface"], dataframe.loyer, label="Surface / Loyer", color="red")
plt.plot(xsample, predicted, color="blue", label="Predicted")
plt.legend()
plt.title("Paris")
plt.show()

# print(model.predict(np.array([[20]])))
# print(20 * model.coef_ + model.intercept_)

