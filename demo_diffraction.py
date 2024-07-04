import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.neighbors as n
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
print(dataframe.describe())

y = dataframe.category
x = dataframe.drop("category", axis=1)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)


plt.matshow(dataframe.corr())
plt.show()

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = n.KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict(xtest)






