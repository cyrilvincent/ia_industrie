import pickle

import numpy as np
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.neighbors as nn
import sklearn.ensemble as tree
import sklearn.svm as svm
import pickle
import sklearn.svm as svm
import sklearn.ensemble as rf
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.neural_network as neural
import tensorflow as tf
import matplotlib.pyplot as plt


np.random.seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"] # 60000
    x_test, y_test = f["x_test"], f["y_test"] # 10000
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding="same")) # 28*28*16
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # 14,14,16

model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28))) # 10*10*16
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # 5*5*16

model.add(tf.keras.layers.Flatten()) # 160
model.add(tf.keras.layers.Dense(160, activation="relu"))
model.add(tf.keras.layers.Dense(80, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print(score)

model.save("data/mnist/mnist.h5")

predicted = model.predict(x_test)
print(predicted)


# Gestion des erreurs
# on récupère les données mal prédites
predicted = predicted.argmax(axis=1)
misclass = (y_test.argmax(axis=1) != predicted)
x_test = x_test.reshape((-1, 28, 28))
misclass_images = x_test[misclass,:,:]
misclass_predicted = predicted[misclass]

# on sélectionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=12)

# on affiche les images et les prédictions (erronées) associées à ces images
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()
