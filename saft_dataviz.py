import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/saft/BMM5/BMM5-CAPA-20230216-100330-DATA_MODULE_1_0.csv", index_col="DATE", delimiter=";", skip_blank_lines=2)
print(dataframe.head())

plt.subplot(231)
plt.plot(dataframe[" CellPackVoltage1"])
plt.subplot(232)
plt.plot(dataframe["CellPackVoltage2"])
plt.subplot(233)
plt.plot(dataframe["CellPackVoltage3"])
plt.subplot(234)
plt.plot(dataframe["ModuleTemperature1"])
plt.subplot(235)
plt.plot(dataframe["DuctTemperature1"])
plt.subplot(236)
plt.plot(dataframe["Ltc6811DieTemperature1"])
plt.show()
