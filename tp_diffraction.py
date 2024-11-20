import pandas as pd

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
# y = category 1 = ko, 0 = ok
# x = tout sauf y, 30 colonnes

# 0 describe()
# 1 Créer x et y
# 2 train_test_split => jeux de train + test
# 3 Créer le modèle linear_regression
# 4 fit
# 5 score => mauvais
