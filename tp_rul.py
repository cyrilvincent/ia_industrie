import pandas as pd
import sklearn.preprocessing as pp
import tensorflow as tf
import sklearn.model_selection as ms
import numpy as np

dataframe = pd.read_csv("data/rul/Battery_RUL.csv", index_col="Cycle_Index")

y = dataframe.RUL
x = dataframe.drop("RUL", axis=1)

np.random.seed(42)
x_train, x_test, y_train, y_test = ms.train_test_split(x, y)


scaler = pp.RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
yscaler = pp.MinMaxScaler()
yscaler.fit(y_train)
y_train = yscaler.transform(y_train)
y_test = yscaler.transform(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
  ])
model.compile(loss="mse", metrics=["accuracy"])
trained = model.fit(x_train, y_train, epochs=20,validation_data=(x_test, y_test))
print(model.summary())
score = model.evaluate(x_test, y_test)
print(score)

ypred = model.predict(x_test)
ypred = yscaler.inverse_transform(ypred)
print(ypred)
