import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_quartiles_stats(q_df):
    stats = {}

    for column in q_df:

        try:
            quartiles = pd.qcut(q_df[column], 4, duplicates='drop')

            labels = [f'Q{i+1}' for i in range(len(quartiles.cat.categories))]

            q_df[f'f {column}_quartile'] = pd.qcut(q_df[column], len(labels),labels = labels, duplicates='drop')


            quartile_stats = q_df.groupby(f'f {column}_quartile')[column].agg(['max', 'min', 'mean', 'var'])
            stats[column] = quartile_stats

            print(f"\nStatistics for {column}:")
            print(quartile_stats)
        except ValueError as e:
            print(f"Error processing {column}: {e}")

    return stats

df = pd.read_csv("Train_data.csv") #Data fields

Types = df.dtypes
fields = df.columns

numeric_df = df.select_dtypes(include=[np.number])

missing_or_inf = numeric_df.isna() | np.isinf(numeric_df)
missing_or_inf_summary = missing_or_inf.sum()

categories = df.nunique()

Max_f = numeric_df.max() #numeric cuz getting the no numeric doesn't seem beneficial
Min_f = numeric_df.min() #numeric cuz getting the no numeric doesn't seem beneficial
Av_f = numeric_df.mean() #numeric to avoid errors
Var_f = numeric_df.var() #numeric to avoid errors

Q_stats = calc_quartiles_stats(numeric_df)


print("Data Fields:\n", list(fields)) #1- a
print("Data Types:\n", Types) #1- b
print("Missing Values: \n", missing_or_inf_summary) #1- c
print("Categories: \n", categories) #1- d
print("Max Value: \n", Max_f) #1- e MAX
print("Min Value: \n", Min_f) #1- e MIN
print("Av Value:\n", Av_f) #1- e Average
print("Var Value:\n", Var_f) #1- e Variance
#print("Q Stats:\n", Q_stats) #1- f

