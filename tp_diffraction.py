import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import pandas as pd

dataframe = pd.read_csv("data/diffraction/data.csv")

y = dataframe.category
x = dataframe.drop(["category", "id"], axis=1)
print(x.columns)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

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