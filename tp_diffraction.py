import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataframe = pd.read_csv("data/diffraction/data.csv")

y = dataframe.category
x = dataframe.drop(["category", "id"], axis=1)
print(x.columns)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# Scaling - Normalisation
# (x - avg) / std
scaler = pp.StandardScaler()
scaler.fit(xtrain) # Calcul de avg et std
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
print(scaler.inverse_transform(xtest))

# for k in range(3,14,2):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     ypred = model.predict(xtest)
#     score = model.score(xtest, ytest)
#     print(k, score)

# model = nn.KNeighborsClassifier(n_neighbors=11)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict([[11.52,18.75,73.34,409,0.09524,0.05473,0.03036,0.02278,0.192,0.05907,0.3249,0.9591,2.183,23.47,0.008328,0.008722,0.01349,0.00867,0.03218,0.002386,12.84,22.47,81.81,506.2,0.1249,0.0872,0.09076,0.06316,0.3306,0.07036]])
print(ypred)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                out_file="data/diffraction/tree.dot",
                feature_names=x.columns,
                class_names=["0", "1"],
                )

with open("data/diffraction/rf.pickle", "wb") as f:
    pickle.dump(model, f)
