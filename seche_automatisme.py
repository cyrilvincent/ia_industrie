import pandas as pd

dataframe = pd.read_excel("data/seche/Automatisme2024-11-201740.xlsx", skiprows=[1])
dataframe.rename( columns={'Unnamed: 0': 't'}, inplace=True )
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
dataframe.set_index("t")
print(dataframe.head(5))