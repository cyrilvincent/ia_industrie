import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_00001 = pd.read_csv("data/nasa/data/00001.csv")
print(df_00001.head())

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
columns = df_00001.columns

# Plot each parameter against time
sns.lineplot(x='Time', y=columns[0], data=df_00001, ax=axes[0, 0])
sns.lineplot(x='Time', y=columns[1], data=df_00001, ax=axes[0, 1])
sns.lineplot(x='Time', y=columns[2], data=df_00001, ax=axes[0, 2])
sns.lineplot(x='Time', y=columns[3], data=df_00001, ax=axes[1, 0])
sns.lineplot(x='Time', y=columns[4], data=df_00001, ax=axes[1, 1])

# Remove empty subplot
fig.delaxes(axes[1, 2])
plt.suptitle('Discharging Parameters vs. Time')

plt.tight_layout()
plt.show()

charge_df = pd.read_csv("data/nasa/charge.csv")


def plot_test_data(df, profile="charge"):
    if profile == 'charge':
        plt.figure(figsize=(10, 4))
        plt.scatter(df.Time, df.Voltage_measured, s=1, color='b', label='Voltage_measured')
        plt.scatter(df.Time, df.Current_measured, s=1, color='r', label='Current_measured')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.scatter(df.Time, df.Voltage_charge, s=1, color='b', label='Voltage_charge')
        plt.scatter(df.Time, df.Current_charge, s=1, color='r', label='Current_charge')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.scatter(df.Time, df.Temperature_measured, s=1, color='k', label='Temperature_measured')
        plt.legend()
        plt.show()
    elif profile == 'discharge':
        plt.figure(figsize=(10, 4))
        plt.plot(df.Time, df.Voltage_measured, 'b', label='Voltage_measured')
        plt.plot(df.Time, df.Current_measured, 'r', label='Current_measured')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(df.Time, df.Voltage_load, 'b', label='Voltage_load')
        plt.plot(df.Time, df.Current_load, 'r', label='Current_load')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(df.Time, df.Temperature_measured, 'k', label='Temperature_measured')
        plt.legend()
        plt.show()
    elif profile == 'impedance':
        pass
    else:
        print('No cycle recognized')

plot_test_data(charge_df[charge_df['battery_id']=='B0047'], profile='charge')

