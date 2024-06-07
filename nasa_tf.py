import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy as np
import sklearn.ensemble as rf
import sklearn.neural_network as nn
import pickle
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("data/nasa/charge.csv")
df = df.dropna()
print(df.describe())

train, test = ms.train_test_split(df)
y_train = train["RUL"]
x_train = train.drop([df.columns[0], 'RUL', 'battery_id'], axis=1)
y_test = test["RUL"]
x_test = test.drop([df.columns[0], 'RUL', 'battery_id'], axis=1)
x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape)

model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

sub_sampling = 1
print(f"LSTM Fit with sub-sampling 1/{sub_sampling}")
model.fit(x_train[::sub_sampling], y_train[::sub_sampling], epochs=10, batch_size=32, validation_data=(x_test, y_test))
score = model.evaluate(x_test[::sub_sampling], y_test[::sub_sampling])
print(f"Score: {score*100:.2f}%")

model.save("data/nasa/model_lstm.h5")
preds = model.predict(x_test[::sub_sampling])
plt.scatter(y_test[::sub_sampling], preds, s=1)
plt.plot(y_test[::sub_sampling], y_test[::sub_sampling], color="red")
plt.show()
