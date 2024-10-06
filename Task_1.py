import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_quartiles_stats(q_df):

    for column in q_df:

        try:
            q_df[f'{column}_quartile'] = pd.qcut(q_df[column], 4, labels= False, duplicates='drop')

            print(f"Statistics for {column}:")
            for quartile in range(4):

                quartile_data = q_df[q_df[f'{column}_quartile'] == quartile][column]

                max_val = quartile_data.max()
                min_val = quartile_data.min()
                avg = quartile_data.mean()
                var = quartile_data.var()

                print(f"Quartiles {quartile+1}")
                print(f"Max: {max_val}")
                print(f"Min: {min_val}")
                print(f"Avg: {avg}")
                print(f"Var: {var}")

        except ValueError as e:
            print(f"Error processing {column}: {e}")


df = pd.read_csv("Train_data.csv") #Data fields

#Part 1
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

#print("Data Fields:\n", list(fields)) #1- a
#print("Data Types:\n", Types) #1- b
#print("Missing Values: \n", missing_or_inf_summary) #1- c
#print("Categories: \n", categories) #1- d
#print("Max Value: \n", Max_f) #1- e MAX
#print("Min Value: \n", Min_f) #1- e MIN
#print("Av Value:\n", Av_f) #1- e Average
#print("Var Value:\n", Var_f) #1- e Variance
#calc_quartiles_stats(numeric_df) #1- f Quartiles

#Part 2
expanded_df = pd.get_dummies(df,columns=['class'], prefix='attack')

#print("Expanded attacks:\n", expanded_df) #2

#Part 3
def plot_pdf_pmf(df_3):
    for column in df_3.columns:
        if df_3.dtypes[column] == 'object':
            plt.figure(figsize=(10,5))
            df_3[column].value_counts(normalize=True).plot(kind='bar')
            plt.title("PMF of {column}:")
            plt.xlabel(column)
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

        elif df_3.dtypes[column] in ['int64', 'float64']:

            if df[column].var() == 0:
                print(f"Skipping column {column} due to zero variance.")
                continue #This part was added to avoid the warnings that appeared due to 0 Var.

            plt.figure(figsize=(10,5))
            sns.kdeplot(df_3[column], fill=True)
            plt.title("PDF of {column}:")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.grid(True)
            plt.show()

#plot_pdf_pmf(df)

#Part 4
def plot_cdf(df_4):
    for column in df.columns:
        sorted_data = df_4[column].sort_values(ascending=True)
        cdf = np.arange(1, len(sorted_data)+1)/ len(sorted_data)

        plt.figure(figsize=(10,5))
        plt.plot(cdf, sorted_data, marker='.', linestyle='none')
        plt.title("CDF of {column}:")
        plt.xlabel("Data")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.show()

plot_cdf(df)
