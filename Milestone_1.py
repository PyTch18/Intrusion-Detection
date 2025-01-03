import random

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
            plt.title(f"PMF of {column}:")
            plt.xlabel(column)
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

        elif df_3.dtypes[column] in ['int64', 'float64']:
            if df_3[column].var() == 0:
                print(f"Skipping column {column} due to zero variance.")
                continue #This part was added to avoid the warnings that appeared due to 0 Var.

            plt.figure(figsize=(10,5))
            #sns.kdeplot(df_3[column], fill=True)
            #sns.histplot(df_3[column], kde=True, stat='density',bins = 30)
            plt.title(f"PDF of {column}:")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.grid(True)
            plt.show()

# part 3 has two ways for plotting commented
# use them by removing the # of the desired one
#plot_pdf_pmf(df)

#Part 4
def plot_cdf(df_4):
    for column in df.columns:
        sorted_data = df_4[column].sort_values(ascending=True)
        cdf = np.arange(1, len(sorted_data)+1)/ len(sorted_data)

        plt.figure(figsize=(10,5))
        plt.plot( sorted_data,  cdf, marker='.', linestyle='none')
        plt.title(f"CDF of {column}:")
        plt.xlabel("Data")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.show()

#plot_cdf(df)

#Part_5
def plot_cond_pdf_pmf(df_5):
    for column in df_5.columns:
        if column == 'class':
            continue # TO avoid checking for class field
        condition = df_5['class'].unique()

        for attack in condition:
            df_5_conditioned = df_5[df_5['class'] == attack]

            if df_5.dtypes[column] == 'object':
                plt.figure(figsize=(10,5))
                df_5[column].value_counts(normalize=True).plot(color= 'blue',kind='bar', label= 'Original PMF')
                df_5_conditioned[column].value_counts(normalize=True).plot(color= 'orange', kind='bar', label= f'Conditional for {attack}')
                plt.title(f"PMF of {column} (Original and Conditional for {attack})")
                plt.legend()
                plt.grid(True)
                plt.show()

            elif df_5.dtypes[column] in ['int64', 'float64']:
                if df_5_conditioned[column].var() == 0:
                    print(f"Skipping column {column} due to zero variance.")
                    continue #This part was added to avoid the warnings that appeared due to 0 Var.

                if df_5[column].var() == 0:
                    print(f"Skipping column {column} due to zero variance.")
                    continue  # This part was added to avoid the warnings that appeared due to 0 Var.

                plt.figure(figsize=(10,5))
                sns.kdeplot(df_5[column],color= 'blue', fill=True, label= 'Original PDF')
                sns.kdeplot(df_5_conditioned[column], color= 'orange', fill=True, label= f'Conditional for {attack}')
                plt.title(f"PDF of {column} (Original and Conditional for {attack})")
                plt.legend()
                plt.grid(True)
                plt.show()

#plot_cond_pdf_pmf(df)

#Part 6
def plot_scatter(df_6):
    numeric_columns = list(df_6.select_dtypes(include=['int64', 'float64']).columns) #Getting the numeric data

    random_columns = random.sample(numeric_columns, 2) #choosing any 2 random fields

    plt.figure(figsize=(10,5))
    sns.scatterplot(x=random_columns[0], y=random_columns[1], data= df_6)
    plt.title(f"Scatterplot of {random_columns[0]} and {random_columns[1]}")
    plt.xlabel(random_columns[0])
    plt.ylabel(random_columns[1])
    plt.grid(True)
    plt.show()

#plot_scatter(df)

#Part 7
def joint_pmf(df_71):
    random_columns = random.sample(list(df_71.columns), 2)  # choosing any 2 random fields

    contingency_table = pd.crosstab(df_71[random_columns[0]], df_71[random_columns[1]], normalize=True)

    plt.figure(figsize=(10,5))
    sns.heatmap(contingency_table, annot=True, cmap="Blues", cbar=True)
    plt.title(f"Joint PMF plot for {random_columns[0]} and {random_columns[1]}")
    plt.grid(True)
    plt.show()


def joint_pdf(df_72):
    # Initialize a flag to control the loop
    valid_columns = False

    while not valid_columns:
        # Randomly choose 2 fields
        random_columns = random.sample(list(df_72.columns), 2)

        var1 = df_72[random_columns[0]].var()
        var2 = df_72[random_columns[1]].var()

        # Check for valid variance and unique values
        if var1 == 0 or var2 == 0 or np.isnan(var1) or np.isnan(var2) or np.isinf(var1) or np.isinf(var2):
            print(f"Skipping {random_columns[0]} and {random_columns[1]} due to zero variance or invalid data.")
            continue  # Keep looping to select new columns

        if df_72[random_columns[0]].nunique() < 2 or df_72[random_columns[1]].nunique() < 2:
            print(f"Skipping {random_columns[0]} and {random_columns[1]} due to insufficient unique values.")
            continue  # Keep looping to select new columns

        # Once valid columns are found, stop the loop
        valid_columns = True

    # Plot scatter plot for very low variance
    if var1 < 1e-5 or var2 < 1e-5:
        print(f"Low variance detected in {random_columns[0]} or {random_columns[1]}, using scatter plot.")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=random_columns[0], y=random_columns[1], data=df_72)
        plt.title(f"Joint Scatter plot for {random_columns[0]} and {random_columns[1]}")
        plt.grid(True)
        plt.show()

    # Plot KDE (Kernel Density Estimate) for valid data
    else:
        try:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(x=df_72[random_columns[0]], y=df_72[random_columns[1]], cmap="Blues", fill=True)
            plt.title(f"Joint PDF for {random_columns[0]} and {random_columns[1]}")
            plt.grid(True)
            plt.show()
        except ValueError as e:
            print(f"Skipping KDE plot for {random_columns[0]} and {random_columns[1]} due to error: {e}")

numeric = df.select_dtypes(include=['int64', 'float64'])
#joint_pdf(numeric)

categorical = df.select_dtypes(include=['object', 'category'])
#joint_pmf(categorical)

#Part 8
def plot_joint_cond_pdf_pmf(df_8):
    for column in df_8.columns:
        if column == 'class':
            continue # TO avoid checking for class field
        condition = df_8['class'].unique()

        for attack in condition:
            df_8_conditioned = df_8[df_8['class'] == attack]

            if df_8_conditioned.dtypes[column] == 'object':
                random_columns = random.sample(list(df_8_conditioned.columns), 2)  # choosing any 2 random fields

                contingency_table = pd.crosstab(df_8_conditioned[random_columns[0]], df_8_conditioned[random_columns[1]], normalize=True)

                plt.figure(figsize=(10, 6))
                sns.heatmap(contingency_table, annot=True, cbar=True)
                plt.title(f"Joint PMF plot for {random_columns[0]} and {random_columns[1]} for condition '{attack}'", fontsize=12)
                plt.grid(True)
                plt.show()


            elif df_8_conditioned.dtypes[column] in ['int64', 'float64']:  # For continuous variables (PDF)
                numeric_columns = df_8_conditioned.select_dtypes(include=['int64', 'float64']).columns

                # Randomly choose 2 numeric fields
                random_columns = random.sample(list(numeric_columns), 2)

                var1 = df_8_conditioned[random_columns[0]].var()
                var2 = df_8_conditioned[random_columns[1]].var()

                # Ensure the data has enough variance and no NaN or Inf values
                if var1 == 0 or var2 == 0 or np.isnan(var1) or np.isnan(var2) or np.isinf(var1) or np.isinf(var2):
                    print(f"Skipping {random_columns[0]} and {random_columns[1]} due to zero variance or invalid data.")
                    continue  # Skip the plot if either column has zero variance or invalid data

                # Check for unique values (at least 2 unique values required)
                if df_8_conditioned[random_columns[0]].nunique() < 2 or df_8_conditioned[
                    random_columns[1]].nunique() < 2:
                    print(
                        f"Skipping {random_columns[0]} and {random_columns[1]} due to insufficient unique values.")
                    continue

                # Use scatter plot for very low variance data
                if var1 < 1e-5 or var2 < 1e-5:
                    print(f"Low variance detected in {random_columns[0]} or {random_columns[1]}, using scatter plot.")

                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=random_columns[0], y=random_columns[1], data=df_8_conditioned)
                    plt.title(f"Joint Scatter plot for {random_columns[0]} and {random_columns[1]} (Attack type: '{attack}')",fontsize=12)
                    plt.grid(True)
                    plt.show()

                else:
                    # Plot KDE (Kernel Density Estimate) for joint PDF
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.kdeplot(x=df_8_conditioned[random_columns[0]], y=df_8_conditioned[random_columns[1]],cmap="Blues", fill=True)
                        plt.title(f"Joint PDF for {random_columns[0]} and {random_columns[1]} (Attack type: '{attack}')",fontsize=12)
                        plt.grid(True)
                        plt.show()
                    except ValueError as e:
                        print(f"Skipping KDE plot for {random_columns[0]} and {random_columns[1]} due to error: {e}")
                        continue

# Extremely heavy code
# Approx. 10 minutes runtime
#plot_joint_cond_pdf_pmf(df)

# Part 9
def correlation_heatmap(df_9):
    # Select only numeric columns for correlation calculation
    numeric_df_9 = df_9.select_dtypes(include=['int64', 'float64'])
    # Calculate the correlation matrix
    correlation_matrix = numeric_df_9.corr()
    # Plot the heatmap
    plt.figure(figsize=(24, 16))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap of Numeric Fields")
    plt.grid(True)
    plt.show()

correlation_heatmap(df)

#part 10
def fields_dependent_on_attack(df_10):
    # Step 1: One-hot encode the 'class' column to create binary attack type columns
    df_10 = pd.get_dummies(df_10, columns=['class'], prefix='attack')

    # Step 2: Check for attack type columns created
    attack_columns = [col for col in df_10.columns if col.startswith('attack_')]
    if not attack_columns:
        raise ValueError("No attack type columns found. Check the 'class' column encoding.")

    print("\nAttack type columns identified:", attack_columns)

    # Step 3: Select only numeric columns for correlation calculation, including attack columns
    numeric_df_10 = df_10.select_dtypes(include=['int64', 'float64']).copy()

    # Manually include attack columns in the numeric dataframe
    numeric_df_10[attack_columns] = df_10[attack_columns]

    # Step 4: Calculate the correlation matrix for numeric fields
    correlation_matrix = numeric_df_10.corr()

    # Step 5: Focus on correlations with attack columns (filter only attack columns)
    attack_correlations = correlation_matrix[attack_columns]

    # Step 6: Display the fields most correlated with each attack type
    for attack_col in attack_columns:
        print(f"\nCorrelation of fields with {attack_col}:\n")
        sorted_correlations = attack_correlations[attack_col].sort_values(ascending=False)
        print(sorted_correlations)

    # Step 7: Visualize the correlation matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(attack_correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation of Fields with Attack Types')
    plt.show()

fields_dependent_on_attack(df)