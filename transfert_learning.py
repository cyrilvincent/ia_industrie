import tensorflow.keras as keras

model = keras.models.load_model('vgg16.h5')

for layer in model.layers[:-1]:
    layer.trainable = False

model.add(keras.layers.Dense(3))
model.add(keras.layers.Activation('softmax'))
