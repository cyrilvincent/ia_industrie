import tensorflow as tf # pip install tensorflow
print(tf.__version__)

import sklearn.preprocessing
import tensorflow as tf
import pandas
import numpy as np


tf.random.set_seed(1)
np.random.seed(1)

dataframe = pandas.read_csv("data/diffraction/data.csv", index_col="id")
y = dataframe.category
x = dataframe.drop("category", axis=1)

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(x)
X = scaler.transform(x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

model.compile(loss="binary_crossentropy",metrics=['accuracy'])
# for 2 categories bce ~= cce
model.summary()

history = model.fit(x, y, epochs=20, batch_size=1, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)
print(f"Total accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")