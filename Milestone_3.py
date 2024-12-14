import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

#import the dataframe
df = pd.read_csv("Train_data.csv")

training_df = df.iloc[:int(df.shape[0]*0.7),:]
training_attack = training_df['class']
testing_df = df.iloc[int(df.shape[0]*0.7):, :]
testing_attack = testing_df['class']

def attack_correlation(training_df_full):
    # Encode the 'class' column as numeric (1 for 'anomaly', 0 for 'normal')
    encoded_attack = training_df_full['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

    # Select only numeric columns
    numeric_df_full = training_df_full.select_dtypes(include=['int64', 'float64']).copy()

    # Filter out columns with constant values or all NaNs
    numeric_df_full = numeric_df_full.loc[:, numeric_df_full.nunique() > 1]

    # Calculate correlations of each numeric column with the encoded 'class' column
    correlation_with_attack = numeric_df_full.apply(lambda x: x.corr(encoded_attack))

    # Convert the result to a dictionary where the column name is the key and correlation is the value
    correlation_dict = correlation_with_attack.to_dict()

    # Replace NaN values with 0 to avoid issues with filtering
    correlation_dict = {k: (v if pd.notna(v) else 0) for k, v in correlation_dict.items()}

    return correlation_dict

# getting the correlations for each column and sorting them
weights = attack_correlation(training_df)
sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
#print(weights)
#print(sorted_weights)

def filter_independent_items(pre_filtered_weights):
    independent_items = {}
    for i, (current_key, current_weight) in enumerate(pre_filtered_weights): # Enumerating the dictionary to be processed 
        is_independent = True
        for prev_key in independent_items.keys():
            # Perform a statistical dependency check
            # Class column is excluded since the method is called for the fields which the class is mostly dependent on
            column_data_x = training_df[prev_key]
            column_data_y = training_df[current_key]
            _, p_value = stats.pearsonr(column_data_x, column_data_y)
            if p_value < 0.05:  # Assuming a significance level of 0.05
                is_independent = False
                break
        if is_independent:
            independent_items[current_key] = current_weight

    # Adding the categorical data to the independent items
    categorical_columns = training_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'class':  # Exclude the 'class' column
            # Perform one-hot encoding for the categorical column
            one_hot_encoded = pd.get_dummies(training_df[col], prefix=col, drop_first=True)
            is_independent = True
            for prev_key in independent_items.keys():
                # Check independence against already chosen independent items
                column_data_x = training_df[prev_key]
                for encoded_col in one_hot_encoded.columns:
                    column_data_y = one_hot_encoded[encoded_col]
                    _, p_value = stats.pearsonr(column_data_x, column_data_y)
                    if p_value < 0.05:  # Assuming a significance level of 0.05
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                independent_items[col] = 0  # Assign 0 as a placeholder weight

                # *NEEDS WORK STILL NOT FUNCTIONING RIGHT*
    
    return independent_items

independent_columns = filter_independent_items(sorted_weights)
print(independent_columns)

def conditioned_data(independent_columns_to_be_conditioned):
    condition = training_df['class'].unique()
    anomaly_conditioned_data = {}
    normal_conditioned_data = {}
    for column in independent_columns_to_be_conditioned:
        for value in condition:
            if value == 'anomaly':
                #Collecting the anomaly conditioned values together
                anomaly_conditioned_data[column] = training_df[training_df['class'] == value][column].dropna()
            else:
                # Collecting the normal conditioned data together
                normal_conditioned_data[column] = training_df[training_df['class'] == value][column].dropna()
    return anomaly_conditioned_data, normal_conditioned_data

# Preparing the data for best fitting by storing in two different sets
data_conditioned_anomaly, data_conditioned_normal = conditioned_data(independent_columns)

# Add the best fit methods for the conditioned and non-conditioned data
# Add the final method to calculate the predicts
# Calculate the 3 metrics for the results

selected_df = df.iloc[:,0:41] #Selecting the columns without the class column
training_df = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df = selected_df.iloc[int(df.shape[0]*0.7):, :]