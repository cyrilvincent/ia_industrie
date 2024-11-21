import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
# y = category 1 = ko, 0 = ok
# x = tout sauf y, 30 colonnes
print(dataframe.describe())
dataframe["rnd"] = np.random.rand(569)
y = dataframe.category
x = dataframe.drop("category", axis=1)

# 0 describe()
# 1 Créer x et y
# 2 train_test_split => jeux de train + test + seed
# 3 Créer le modèle linear_regression
# 4 fit
# 5 score => mauvais

scaler = pp.StandardScaler()
scaler.fit(x)
xscaler = scaler.transform(x)


np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(xscaler, y, train_size=0.8, test_size=0.2)
# model = lm.LinearRegression()
# for k in range(3,12,2):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     train_score = model.score(xtrain, ytrain)
#     test_score = model.score(xtest, ytest)
#     print(f"k={k}, training score: {train_score * 100:.0f}%, test score: {test_score * 100:.0f}%")
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)
print(f"training score: {train_score * 100:.0f}%, test score: {test_score * 100:.0f}%")

print(model.feature_importances_)

predicted = model.predict(xtest)
print(metrics.confusion_matrix(ytest, predicted))
print(metrics.classification_report(ytest, predicted))

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/diffraction/tree.dot',
                 feature_names = x.columns,
                 class_names = ["0", "1"],
                 rounded = True, proportion = False,
                 precision = 2, filled = True)

