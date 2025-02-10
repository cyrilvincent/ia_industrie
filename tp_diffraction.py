import sklearn.neighbors as nn
import sklearn.model_selection as ms
import pandas as pd

dataframe = pd.read_csv("data/diffraction/data.csv")

y = dataframe.category
x = dataframe.drop(["category", "id"], axis=1)
print(x.columns)

# TODO train + test
# train
# predire
# score
