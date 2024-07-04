import sklearn
import sklearn.linear_model as lm # pip install scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

print(sklearn.__version__)
print(datetime.datetime.now().isoformat())

# 0 Chargement des data
dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

# 1 Création du modèle
model = lm.LinearRegression()

# 2 Fit
model.fit(x, y)
print(model.coef_, model.intercept_)

# 3 Predict
ypred = model.predict(x)

# 4 Scoring
score = model.score(x, y)
print(score)

# 5 Display
plt.scatter(x, y)
plt.plot(x, ypred, color="red")
plt.show()