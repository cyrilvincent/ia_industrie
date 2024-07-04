import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

m1 = np.array([[1,2],[3,4]]) # Row first
print(m1)
print(m1.shape)
m2 = np.array([[5,6],[7,8]])
print(m1+m2)

v1 = np.arange(12)
print(v1.shape)
m34 = v1.reshape(2,3,2)
print(m34)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())
dataframe.values # => numpy
loyer_per_m2 = dataframe.loyer / dataframe["surface"]
print(loyer_per_m2.describe())

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()

