import pandas as pd
import tqdm
import os

metadata_df = pd.read_csv("metadata.csv")
print(metadata_df.head())
print(metadata_df[metadata_df['battery_id'] == 'B0048'].head(10))

discharge_df = pd.DataFrame()
charge_df = pd.DataFrame()

folder_path = 'data/'

for index, row in tqdm.tqdm(metadata_df.iterrows()):
    file_name = row['filename']
    battery_id = row['battery_id']
    charge_cycle, discharge_cycle = 0, 0
    file_path = os.path.join(folder_path, file_name)
    try:
        file_df = pd.read_csv(file_path)
        file_df['battery_id'] = battery_id
        if row['type'] != 'impedance':
            final_time = file_df.Time.iloc[-1]
            file_df['RUL'] = final_time - file_df['Time']
        if row['type'] == 'discharge':
            discharge_cycle += 1
            file_df['cycle'] = discharge_cycle
            discharge_df = pd.concat([discharge_df, file_df], ignore_index=True)
        elif row['type'] == 'charge':
            charge_cycle += 1
            file_df['cycle'] = charge_cycle
            charge_df = pd.concat([charge_df, file_df], ignore_index=True)
    except Exception as ex:
        print(f"Error on file {file_path}: {ex}")

print("Discharge DataFrame:")
print(discharge_df.head())
discharge_df.to_csv("data/nasa/discharge.csv")
print("Charge DataFrame:")
print(charge_df.head())
charge_df.to_csv("data/nasa/charge.csv")