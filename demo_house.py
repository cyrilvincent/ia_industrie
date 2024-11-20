import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe.loyer / dataframe["surface"]
print(dataframe.describe())
dataframe.to_json("data/house/house.json")

# 1 data
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

# 2 = créer le modèle
model = lm.LinearRegression()

# 3 = Fit = Apprentissage
model.fit(x, y)

# 4 Prédiction
predicted = model.predict(x)

# 5 Display
plt.scatter(dataframe["surface"], dataframe.loyer, label="Surface / Loyer", color="red")
plt.plot(dataframe.surface, predicted, color="blue", label="Predicted")
plt.legend()
plt.title("Paris")
plt.show()

