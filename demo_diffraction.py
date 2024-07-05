import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.neighbors as n
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.ensemble as rf
import sklearn.metrics as m
import sklearn.neural_network as nn

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
print(dataframe.describe())

y = dataframe.category
x = dataframe.drop("category", axis=1)
xorig = x

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)


# plt.matshow(dataframe.corr())
# plt.show()

np.random.seed(42)

scaler = pp.StandardScaler()
scaler.fit(x) # Calcul moy et std pour chacune des colonnes
x = scaler.transform(x) # (x - moy) / std
# scaler.inverse_transform(x)


xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)





# model = lm.LinearRegression()
# model = n.KNeighborsClassifier(n_neighbors=5)
# model = rf.RandomForestClassifier(max_depth=5)
model = nn.MLPClassifier(hidden_layer_sizes=(30,10))
model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict(xtest)

print(m.classification_report(ytest, ypred))

print(m.confusion_matrix(ytest, ypred))

# print(model.feature_importances_)
# plt.bar(xorig.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()
#
# from sklearn.tree import export_graphviz
# export_graphviz(model.estimators_[0],
#                  out_file='data/diffraction/tree.dot',
#                  feature_names = xorig.columns,
#                  class_names = ["0", "1"],
#                  rounded = True, proportion = False,
#                  precision = 2, filled = True)






