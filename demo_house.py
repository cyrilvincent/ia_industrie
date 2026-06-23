import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import sklearn.neighbors as nn

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")


# 1 data
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# 2 = créer le modèle
model = lm.LinearRegression()

# f(x) = a * x + b
# f(x) = 3 * x + 2
# f(x) = ax² + bx + c

# model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
# degre = 1
# model = pipe.make_pipeline(pp.PolynomialFeatures(degre), lm.Ridge())
# for k in range(3,12,2):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     print(k, model.score(xtest, ytest))

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
plt.scatter(dataframe["surface"], dataframe["loyer"], label="Surface / Loyer", color="red")
plt.plot(xsample, predicted, color="blue", label="Predicted")
plt.legend()
plt.title("Paris")
plt.show()

# print(model.predict(np.array([[20]])))
# print(20 * model.coef_ + model.intercept_)

