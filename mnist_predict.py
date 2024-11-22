import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"] # 60000
    x_test, y_test = f["x_test"], f["y_test"] # 10000
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


model = tf.keras.models.load_model("data/mnist/mnist.h5")
predicted = model.predict(x_test)
print(predicted)
y_test = tf.keras.utils.to_categorical(y_test)


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
