import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

print(pd.__version__)

# 0 Load les data
dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)
print(dataframe.describe())

print(dataframe.loyer / dataframe["surface"])

# 1 Création du modèle
model = lm.LinearRegression()


# X = training set
# Y = Label
x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# 2 Training
model.fit(xtrain,ytrain)

# 3 Prediction
ypred = model.predict(xtest)
score = model.score(xtest, ytest)
print(score)

# 5 Display
plt.scatter(xtest, ytest, color="blue")
plt.plot(xtest, ypred, color="red")
plt.show()

# 1000 data
# testset = 200
# log(200)~=2
# Nombre de chiffre significatif log-1 = 2-1 = 1
# 50000 data
# testset = 10000
# log(10000)=4
# 3 chiffres siginificatif
