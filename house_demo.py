import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)
print(dataframe.describe())
# 0 Load les data
print(dataframe.loyer / dataframe["surface"])

# 1 Création du modèle
model = lm.LinearRegression()


# X = training set
# Y = Label
x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

# 2 Training
model.fit(x,y)

# 3 Prediction
ypred = model.predict(x)
score = model.score(x, y)
print(score)

# 5 Display
plt.scatter(x, y, color="blue")
plt.plot(x, ypred, color="red")
plt.show()
