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

np.random.seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"] # 60000
    x_test, y_test = f["x_test"], f["y_test"] # 10000
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# model = nn.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()
model.fit(x_train, y_train)

print(model.score(x_test, y_test), model.score(x_train, y_train))
predicted = model.predict(x_test)

print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


matrix = model.feature_importances_.reshape(28, 28)
plt.imshow(matrix)
plt.show()

images = x_test.reshape((-1, 28, 28))



# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])

plt.show()

# Gestion des erreurs
# on récupère les données mal prédites
misclass = (y_test != predicted)
misclass_images = images[misclass,:,:]
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



