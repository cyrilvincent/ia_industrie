import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt

print("Hello World")
print(sys.version)
print(sklearn.__version__)
print(np.__version__)

a1 = np.array([1,2,3,4])
a2 = np.array([5,6,7,8])
print(a1 + a2)

v = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
print(len(v))
y = np.sin(v)
print(y)

plt.plot(v, y)
plt.show()
