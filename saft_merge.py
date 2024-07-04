import pandas as pd
import tqdm
import os

dataframe = pd.DataFrame()

path = "data/saft/BMM5"
l = os.listdir(path)
for f in l:
    if f.endswith(".csv") and f != "merge.csv":
        file_path = os.path.join(path, f)
        try:
            print(f"Load {file_path}")
            df = pd.read_csv(file_path, delimiter=";", skip_blank_lines=2, index_col=" ELAPSED_SECONDS")
            dataframe = pd.concat([dataframe, df], ignore_index=True)
        except Exception as ex:
            print(f"Error on file {file_path}: {ex}")
print("Writing to merge.csv")
dataframe.to_csv(f"{path}/merge.csv")
