import array

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from distfit import distfit
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns

#import the dataframe
df = pd.read_csv("Train_data.csv")
#choose all the dataframe except the class column
selected_df = df.iloc[:,0:41]
#selected_df = df.drop(columns=['class'])
#display(selected_df.to_string())

training_df = df.iloc[:int(df.shape[0]*0.7),:]
testing_df = df.iloc[int(df.shape[0]*0.7):, :]

#to determine the best distribution we will use distfit as it trys the data on 89 different distributions
def best_fit_distribution(df1):
    dist = distfit()
    for column in df1.columns:
        if df1.dtypes[column] in ['int64', 'float64']:
            print(f"\nFitting distribution for column: {column}")
            try:
                # Filter data to exclude outliers (5th to 95th percentiles)
                lower_bound = np.percentile(df1[column].dropna(), 5)
                upper_bound = np.percentile(df1[column].dropna(), 95)
                filtered_data = df1[(df1[column] >= lower_bound) & (df1[column] <= upper_bound)][column]

                # Additional checks for sufficient unique values and variance
                if filtered_data.nunique() < 10:
                    print(f"Skipping '{column}' due to insufficient unique values after filtering.")
                    continue  # Skip this column if there are too few unique values

                if filtered_data.var() < 1e-5:
                    print(f"Skipping '{column}' due to low variance.")
                    continue  # Skip this column if variance is too low



                dist.fit_transform(filtered_data.dropna())  # Drop NaN values if any

                # Plot the best-fit distribution
                dist.plot()
                plt.show()  # Ensure the plot is displayed

                # Print the best-fitting model for the column
                print(f"Best distribution for '{column}': {dist.model}")

            except Exception as e:
                print(f"Error fitting distribution for column '{column}': {e}")
                continue  # Skip to the next column if an error occurs

#best_fit_distribution(df)

def z_score(df2, thresholds1):
    result = {}

    for column in df2.columns:
        # Skip non-numeric columns and the target 'class' column
        if df2.dtypes[column] in ['int64', 'float64'] and column != 'class':
            # Calculate Z-scores for the column
            mean = df2[column].mean()  # Mean for the column
            std = df2[column].std()  # Standard deviation for the column
            z_scores = (df2[column] - mean) / std  # Z-scores

            # Split Z-scores based on 'class' column
            normal_scores = z_scores[df2['class'] == 'normal']
            anomaly_scores = z_scores[df2['class'] == 'anomaly']

            result[column] = {}

            # Check anomalies at each threshold
            for threshold in thresholds1:
                # Find anomalies where Z-score exceeds the threshold
                normal_anomalies = normal_scores[abs(normal_scores) > threshold]
                detected_anomalies = anomaly_scores[abs(anomaly_scores) > threshold]

                # Store counts of anomalies for each threshold
                result[column][threshold] = {
                    'normal_count_above_threshold': len(normal_anomalies),
                    'anomaly_count_above_threshold': len(detected_anomalies)
                }

                # Print the results for this feature and threshold
                print(f"Feature '{column}' with threshold {threshold}:")
                print(f"  Normal count above threshold: {len(normal_anomalies)}")
                print(f"  Anomaly count above threshold: {len(detected_anomalies)}\n")

    return result

# Example usage
thresholds = [1.5, 2.0, 2.5, 3.0]
#anomaly_results = z_score(df, thresholds)

def performance_metrics(df3):
    actual= (df3['class'] == 'anomaly').astype(int) # if the class data is an anomaly it will be stored as 1 and if the data
                                                    # is normal it will be stored as 0
    predicted = array.array('i')
    for rows in df3.rows:
        error = 0
        for column in df3.columns:
            if df.dtypes[column] in ['int64', 'float64'] and column != 'class':
                x_value = df3[column,rows]
                if z_score(x_value, thresholds) > thresholds[0]:
                    error += 1
        if error >= 6:
            predicted.append(1) # anomaly
        else:
            predicted.append(0) # normal
    matrix = confusion_matrix(actual,predicted)
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

#performance_metrics(df)

# Task 2 part (III)
# Summarize best-fit PDF for numerical columns
def document_best_fit_pdf(df91):
    dist = distfit()
    result_summary = {}
    for column in df91.select_dtypes(include=[np.number]).columns:
        if df91[column].nunique() < 10:
            print(f"Skipping '{column}' due to insufficient unique values.")
            continue
        if df91[column].var() < 1e-5:
            print(f"Skipping '{column}' due to low variance.")
            continue
        try:
            lower_bound = np.percentile(df91[column].dropna(), 2)
            upper_bound = np.percentile(df91[column].dropna(), 98)
            filtered_data = df91[(df91[column] >= lower_bound) & (df91[column] <= upper_bound)][column]
            dist.fit_transform(filtered_data.dropna())
            best_distribution = dist.model
            result_summary[column] = {
                'best_fit_distribution': best_distribution['name'],
                'params': best_distribution['params']
            }
        except Exception as e:
            print(f"Error fitting distribution for '{column}': {e}")
    return result_summary

# Summarize PMF data for categorical columns
def document_pmf_data(df92):
    pmf_summary = {}
    for column in df92.select_dtypes(include=['object']).columns:
        try:
            pmf = df92[column].value_counts(normalize=True)
            pmf_summary[column] = pmf.to_dict()  # Store PMF as dictionary for future reference
        except Exception as e:
            print(f"Error calculating PMF for '{column}': {e}")
    return pmf_summary

# Document results for both numerical and categorical columns
def document_analysis_results(df93):
    # Summarize best-fit distributions for numerical columns
    numerical_summary = document_best_fit_pdf(df93)
    # Summarize PMF data for categorical columns
    categorical_summary = document_pmf_data(df93)
    return numerical_summary, categorical_summary


def print_summary(numerical_summary, categorical_summary):
    print("\nPDF Summary (Numerical Columns):")
    for column, info in numerical_summary.items():
        print(f"  - Column: {column}")
        print(f"    Best-Fit Distribution: {info['best_fit_distribution']}")
        print(f"    Parameters: {info['params']}")

    print("\nPMF Summary (Categorical Columns):")
    for column, pmf in categorical_summary.items():
        print(f"  - Column: {column}")
        print("    PMF:")
        for value, probability in pmf.items():
            print(f"      {value}: {probability:.4f}")

ns,cs = document_analysis_results(df)
print_summary(ns, cs)