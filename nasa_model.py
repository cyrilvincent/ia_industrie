import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy as np
import sklearn.ensemble as rf
import sklearn.neural_network as nn
import pickle

np.random.seed(42)
df = pd.read_csv("data/nasa/charge.csv")
df = df.dropna()
print(df.describe())

train, test = ms.train_test_split(df)
y_train = train["RUL"]
x_train = train.drop([df.columns[0], 'RUL', 'battery_id'], axis=1)
y_test = test["RUL"]
x_test = test.drop([df.columns[0], 'RUL', 'battery_id'], axis=1)
print(x_train.shape)

model = rf.RandomForestRegressor()
sub_sampling = 1000
print(f"RF fit with sub-sampling 1/{sub_sampling}")
model.fit(x_train[::sub_sampling], y_train[::sub_sampling])
with open("data/nasa/charge_rf.pickle", "wb") as f:
    pickle.dump(model, f)
score = model.score(x_test, y_test)
print(f"Score: {score*100:.2f}%")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print(f"{x_train.columns[indices[f]]}: {importances[indices[f]]*100:.2f}%")

preds = model.predict(x_test)
plt.scatter(y_test, preds, s=1)
plt.plot(y_test[::sub_sampling], y_test[::sub_sampling], color="red")
plt.show()

model = nn.MLPRegressor(hidden_layer_sizes=(10, 10, 10))
sub_sampling = 10
print(f"MLP fit with sub-sampling 1/{sub_sampling}")
model.fit(x_train[::sub_sampling], y_train[::sub_sampling])
with open("data/nasa/charge_mlp.pickle", "wb") as f:
    pickle.dump(model, f)
score = model.score(x_test, y_test)
print(f"Score: {score*100:.2f}%")
preds = model.predict(x_test)
plt.scatter(y_test, preds, s=1)
plt.plot(y_test[::sub_sampling], y_test[::sub_sampling], color="red")
plt.show()



