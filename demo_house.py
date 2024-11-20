import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe.loyer / dataframe["surface"]
print(dataframe.describe())
dataframe.to_json("data/house/house.json")


plt.scatter(dataframe["surface"], dataframe.loyer, label="Surface / Loyer", color="red")
plt.legend()
plt.title("Paris")
plt.show()

