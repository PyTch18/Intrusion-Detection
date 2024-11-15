import numpy as np
import pandas as pd
from distfit import distfit
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

#import the dataframe
df = pd.read_csv("Train_data.csv")
#selected_df = df.drop(columns=['class'])
#display(selected_df.to_string())

training_df = df.iloc[:int(df.shape[0]*0.7),:]
training_attack = training_df['class']
testing_df = df.iloc[int(df.shape[0]*0.7):, :]
testing_attack = testing_df['class']


#Task 1_2
def attack_correlation(df_full):
    # Encode the 'class' column as numeric (1 for 'anomaly', 0 for 'normal')
    df_full['class_encoded'] = df_full['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

    # Select only numeric columns
    numeric_df_full = df_full.select_dtypes(include=['int64', 'float64']).copy()

    # Calculate correlations of each numeric column with the 'class_encoded' column
    correlation_with_attack = numeric_df_full.apply(lambda x: x.corr(df_full['class_encoded']))

    # Drop the temporary 'class_encoded' column
    df_full.drop(columns=['class_encoded'], inplace=True)

    # Convert the result to a dictionary where the column name is the key and correlation is the value
    correlation_dict = correlation_with_attack.to_dict()

    # Replace NaN values with 0 to avoid issues with filtering
    correlation_dict = {k: (v if pd.notna(v) else 0) for k, v in correlation_dict.items()}

    return correlation_dict

weights = attack_correlation(training_df)

#choose all the dataframe except the class column

selected_df = df.iloc[:,0:41]
training_df = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df = selected_df.iloc[int(df.shape[0]*0.7):, :]



def z_score(df2, threshold,weights2):
    # Get the weights (correlations) for numeric columns
    print(f"Weights: {weights2}")  # Debugging: Print weights
    predictions = []

    # Get the numeric columns
    numeric_columns = df2.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numeric Columns: {list(numeric_columns)}")  # Debugging: Print numeric columns

    # Filter out columns with zero correlation
    valid_columns = [col for col in numeric_columns if weights2.get(col, 0) != 0]
    print(f"Valid Columns (Non-zero Correlation): {valid_columns}")  # Debugging: Print valid columns

    if not valid_columns:
        raise ValueError("No columns have a valid (non-zero) correlation")

    # Create a mapping of valid column names to their corresponding weights
    column_weights = {col: weights2[col] for col in valid_columns}
    print(f"Column Weights Mapping (Valid Columns): {column_weights}")  # Debugging: Print column-weights mapping

    # Loop through each row to calculate weighted Z-scores and determine predictions
    for index, row in df2.iterrows():
        total_weighted_z_score = 0  # Sum of weighted Z-scores for this row

        for column in valid_columns:
            if df2.dtypes[column] in ['int64', 'float64']:
                mean = df2[column].mean()
                std = df2[column].std()

                # Avoid division by zero
                if std == 0:
                    continue

                # Calculate Z-score
                z_score_2 = (row[column] - mean) / std

                # Apply weight
                weight = column_weights.get(column, 0)  # Default weight is 0 if not in column_weights
                weighted_z_score = z_score_2 * weight
                total_weighted_z_score += weighted_z_score

        # Determine the prediction based on total weighted Z-score
        anomaly_threshold = threshold  # Adjust this logic as needed
        if total_weighted_z_score > anomaly_threshold:
            predictions.append(1)  # The 1 here is indicating anomaly
        else:
            predictions.append(0)  # The 0 here is indicating normal

    return predictions

# Example usage
thresholds = [1.5, 2.0, 2.5, 3.0]
# 1.5 gave the best recall and accuracy so smaller thresholds were tried till 0.5
# 0.5 gave the best results in terms of recall and accuracy
# smaller values has higher recall but lower accuracy and precision
#predict = z_score(training_df, 0.5, weights)
#print(predict)

# For Accuracy and recall threshold of 1.5 is better
# For precision a threshold of 3.0 is the better

# threshold of 0.5 is the best with max accuracy and recall
testing_predict = z_score(testing_df, 0.5, weights)

def performance_metrics(attack_3, pridect_3):
    attack_3 = attack_3.apply(lambda x: 1 if x == 'anomaly' else 0)
    matrix = confusion_matrix(attack_3,pridect_3)
    if matrix.shape == (2,2):
        tn, fp, fn, tp = matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
    else:
        print("wrong data set")

#performance_metrics(training_attack, predict)

performance_metrics(testing_attack, testing_predict)
