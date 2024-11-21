import tensorflow as tf
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


print(tf.__version__)

dataframe = pd.read_csv("data/diffraction/data.csv", index_col="id")
# y = category 1 = ko, 0 = ok
# x = tout sauf y, 30 colonnes
print(dataframe.describe())
y = dataframe.category
x = dataframe.drop("category", axis=1)

scaler = pp.StandardScaler()
scaler.fit(x)
xscaler = scaler.transform(x)

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(xscaler, y, train_size=0.8, test_size=0.2)

tf.random.set_seed(0)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation="relu", input_shape=(x.shape[1],)))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="mse", metrics=["accuracy"])
model.summary()

model.fit(xtrain, ytrain, epochs=20, validation_data=(xtest, ytest)) #
score = model.evaluate(x, y)
print(score)




