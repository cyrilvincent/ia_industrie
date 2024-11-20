import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
# y = category 1 = ko, 0 = ok
# x = tout sauf y, 30 colonnes
print(dataframe.describe())
y = dataframe.category
x = dataframe.drop("category", axis=1)

# 0 describe()
# 1 Créer x et y
# 2 train_test_split => jeux de train + test + seed
# 3 Créer le modèle linear_regression
# 4 fit
# 5 score => mauvais

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
model = lm.LinearRegression()
model.fit(xtrain, ytrain)
train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)
print(f"Training score: {train_score * 100:.0f}%, test score: {test_score * 100:.0f}%")

