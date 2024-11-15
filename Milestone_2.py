import array

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from distfit import distfit
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.stats as stats

#import the dataframe
df = pd.read_csv("Train_data.csv")
#choose all the dataframe except the class column
selected_df = df.iloc[:,0:41]
#selected_df = df.drop(columns=['class'])
#display(selected_df.to_string())

training_df = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df = selected_df.iloc[int(df.shape[0]*0.7):, :]

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
# Summarize best-fit distributions for numerical columns
def document_best_fit_pdf(df91, class_column='class'):
    distributions = [
        stats.alpha, stats.norm, stats.expon, stats.gamma, stats.pareto, stats.beta, stats.lognorm, stats.weibull_min,
        stats.weibull_max, stats.t, stats.f, stats.chi2, stats.gumbel_r, stats.gumbel_l, stats.dweibull,
        stats.genextreme, stats.uniform, stats.arcsine, stats.cosine, stats.exponnorm, stats.foldcauchy
    ]
    result_summary = {}

    for column in df91.select_dtypes(include=[np.number]).columns:
        if column == class_column or df91[column].nunique() < 10 or df91[column].var() < 1e-5:
            print(f"Skipping '{column}' due to low variance or insufficient unique values.")
            continue

        # Filter extreme values (2nd to 98th percentiles)
        lower_bound = np.percentile(df91[column].dropna(), 2)
        upper_bound = np.percentile(df91[column].dropna(), 98)
        filtered_data = df91[(df91[column] >= lower_bound) & (df91[column] <= upper_bound)][column].dropna()

        # Use the provided best_fit_distribution function to determine the best fit
        try:
            bin_edges = np.linspace(lower_bound, upper_bound, 15)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            best_distribution, best_params, best_mse = best_fit_distribution(filtered_data, bin_centers, distributions)

            if best_distribution:
                result_summary[column] = {
                    'best_fit_distribution': best_distribution.name,
                    'params': best_params,
                    'mse': best_mse
                }
        except Exception as e:
            print(f"Error fitting distribution for '{column}': {e}")
    return result_summary

# Summarize PMF data for categorical columns
def document_pmf_data(df92, class_column='class'):
    pmf_summary = {}
    for column in df92.select_dtypes(include=['object']).columns:
        try:
            pmf_summary[column] = {}
            # Overall PMF
            overall_pmf = df92[column].value_counts(normalize=True).to_dict()
            pmf_summary[column]['overall'] = overall_pmf

            # Class-conditioned PMFs
            for class_value in df92[class_column].unique():
                conditioned_pmf = df92[df92[class_column] == class_value][column].value_counts(normalize=True).to_dict()
                pmf_summary[column][class_value] = conditioned_pmf
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
        print(f"    Mean Squared Error (MSE): {info['mse']:.5f}")
        print("\n")

    print("PMF Summary (Categorical Columns):")
    for column, pmf in categorical_summary.items():
        print(f"\n  - Column: {column}")
        print("    Overall PMF:")
        for value, probability in pmf['overall'].items():
            print(f"      {value}: {probability:.4f}")

        for class_value, class_pmf in pmf.items():
            if class_value == 'overall':
                continue
            print(f"    PMF Conditioned on {class_value}:")
            for value, probability in class_pmf.items():
                print(f"      {value}: {probability:.4f}")
    print("\n")
ns, cs = document_analysis_results(df)
print_summary(ns, cs)