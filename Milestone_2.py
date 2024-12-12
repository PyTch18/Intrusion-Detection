import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from distfit import distfit
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

#import the dataframe
df = pd.read_csv("Train_data.csv")
#choose all the dataframe except the class column
selected_df = df.iloc[:,0:41]
#selected_df = df.drop(columns=['class'])
#display(selected_df.to_string())

#Task 1 Part (I)
training_df = df.iloc[:int(df.shape[0]*0.7),:]
training_attack = training_df['class']
testing_df = df.iloc[int(df.shape[0]*0.7):, :]
testing_attack = testing_df['class']

#Task 1 Part (II)

def attack_correlation(df_full):
    # Encode the 'class' column as numeric (1 for 'anomaly', 0 for 'normal')
    encoded_attack = df_full['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

    # Select only numeric columns
    numeric_df_full = df_full.select_dtypes(include=['int64', 'float64']).copy()

    # Filter out columns with constant values or all NaNs
    numeric_df_full = numeric_df_full.loc[:, numeric_df_full.nunique() > 1]

    # Calculate correlations of each numeric column with the encoded 'class' column
    correlation_with_attack = numeric_df_full.apply(lambda x: x.corr(encoded_attack))

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
    #print(f"Weights: {weights2}")  # Debugging: Print weights
    predictions = []

    # Get the numeric columns
    numeric_columns = df2.select_dtypes(include=['int64', 'float64']).columns
    #print(f"Numeric Columns: {list(numeric_columns)}")  # Debugging: Print numeric columns

    # Filter out columns with zero correlation
    valid_columns = [col for col in numeric_columns if weights2.get(col, 0) != 0]
    #print(f"Valid Columns (Non-zero Correlation): {valid_columns}")  # Debugging: Print valid columns

    if not valid_columns:
        raise ValueError("No columns have a valid (non-zero) correlation")

    # Create a mapping of valid column names to their corresponding weights
    column_weights = {col: weights2[col] for col in valid_columns}
    #print(f"Column Weights Mapping (Valid Columns): {column_weights}")  # Debugging: Print column-weights mapping

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

thresholds = [1.5, 2.0, 2.5, 3.0]

# 1.5 gave the best recall and accuracy so smaller thresholds were tried till 0.5
# 0.5 gave the best results in terms of recall and accuracy
# smaller values has higher recall but lower accuracy and precision


# For Accuracy and recall threshold of 1.5 is better
# For precision a threshold of 3.0 is the better
# threshold of 0.5 is the best with max accuracy and recall

#predict = z_score(training_df, 0.5, weights)
testing_predict = z_score(testing_df, 0.5, weights)

def performance_metrics(attack_3, predict_3):
    attack_3 = attack_3.apply(lambda x: 1 if x == 'anomaly' else 0)
    matrix = confusion_matrix(attack_3, predict_3)
    if matrix.shape == (2,2):
        """True Negative (tn): Correctly predicted normal points.
           False Positive (fp): Normal points wrongly predicted as anomalies.
           False Negative (fn): Anomalies wrongly predicted as normal.
           True Positive (tp): Correctly predicted anomalies."""
        tn, fp, fn, tp = matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) # Accuracy is the ratio of correctly predicted observations to the total observations.
        precision = tp / (tp + fp) # Precision is the ratio of correctly predicted positive observations (anomalies) to the total predicted positive observations.
        recall = tp / (tp + fn) # Recall is the ratio of correctly predicted positive observations (anomalies) to all actual positive observations.
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
    else:
        print("wrong data set")

#performance_metrics(training_attack, predict)

performance_metrics(testing_attack, testing_predict)

#Task 2 part (I)
def calculate_mse(empirical_counts, fitted_pdf):
    """Calculate Mean Squared Error (MSE) between empirical data and fitted PDF."""
    return np.mean((empirical_counts - fitted_pdf) ** 2)


def best_fit_distribution(data, bin_centers, distributions):
    """Find the best-fitting distribution by calculating MSE for each."""
    best_mse = float('inf')
    best_distribution = None
    best_params = None

    # Calculate empirical counts based on bin centers
    empirical_counts, _ = np.histogram(data, bins=len(bin_centers), range=(bin_centers.min(), bin_centers.max()),
                                       density=True)

    for distribution in distributions:
        try:
            # Fit the distribution to data
            params = distribution.fit(data)

            # Calculate the PDF with fitted parameters
            fitted_pdf = distribution.pdf(bin_centers, *params) # *params bta2sem nafsaha 3la 7asab kol distribution

            # Check shapes before calculating MSE
            if len(empirical_counts) != len(fitted_pdf):
                print(
                    f"Shape mismatch: empirical_counts has length {len(empirical_counts)}, fitted_pdf has length {len(fitted_pdf)} for {distribution.name}")
                continue

            # Calculate MSE
            mse = calculate_mse(empirical_counts, fitted_pdf)

            # Update the best distribution if this one has the lowest MSE
            if mse < best_mse:
                best_mse = mse
                best_distribution = distribution
                best_params = params
        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")
            continue

    return best_distribution, best_params, best_mse


def plot_conditional_pdfs(df3, unique_values_threshold=10):
    # Define a list of distributions to test
    distributions = [
        stats.alpha, stats.norm, stats.expon, stats.gamma, stats.pareto, stats.beta, stats.lognorm, stats.weibull_min,
        stats.weibull_max, stats.t, stats.f, stats.chi2, stats.gumbel_r, stats.gumbel_l, stats.dweibull,
        stats.genextreme, stats.uniform , stats.arcsine, stats.cosine, stats.exponnorm, stats.foldcauchy
    ]

    # Select only numerical columns, excluding the 'class' column
    numerical_columns = df3.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if col != 'class']

    # Define conditions
    class_conditions = {
        'Original': df3,
        'Normal': df3[df3['class'] == 'normal'],
        'Anomaly': df3[df3['class'] == 'anomaly']
    }

    # Loop through each numerical column
    for column in numerical_columns:
        # Check if the column has enough unique values
        if df3[column].nunique() < unique_values_threshold:
            print(f"Skipping '{column}' due to low unique values.")
            continue

        # Calculate the IQR and determine if we should adjust the x-axis range
        q1, q3 = np.percentile(df3[column].dropna(), [25, 75])
        iqr = q3 - q1
        if df3[column].max() > q3 + 10 * iqr or df3[column].min() < q1 - 10 * iqr:
            lower_bound, upper_bound = np.percentile(df3[column].dropna(), [0 , 97])
        else:
            lower_bound, upper_bound = df3[column].min(), df3[column].max()

        # Set up plot with restricted x-axis range for extreme data
        plt.figure(figsize=(10, 6))
        plt.title(f'PDF of {column} with Best Fit (Original, Normal, and Anomaly)')

        # Plot PDFs and find best fit for each condition
        colors = {'Original': 'blue', 'Normal': 'green', 'Anomaly': 'red'}
        for condition_name, condition_data in class_conditions.items():
            data_conditioned = condition_data[column].dropna()

            # Plot the empirical PDF for the condition
            sns.histplot(data_conditioned, kde=True , stat='density', label=condition_name,
                         color=colors[condition_name], bins=15, element='step')

            # Calculate best-fitting distribution using MSE
            bin_edges = np.linspace(lower_bound, upper_bound, 15)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

            best_distribution, best_params, best_mse = best_fit_distribution(data_conditioned, bin_centers,
                                                                             distributions)

            # Plot the best-fitting PDF as a dotted line
            if best_distribution:
                fitted_pdf = best_distribution.pdf(bin_centers, *best_params)
                plt.plot(bin_centers, fitted_pdf, linestyle='--', color=colors[condition_name],
                         label=f'Best Fit ({condition_name}): {best_distribution.name} (MSE={best_mse:.5f})')

        # Display plot settings
        plt.xlim(lower_bound, upper_bound)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend(title="Condition")
        plt.grid(True)
        plt.show()

plot_conditional_pdfs(df) #this is the most correct thing I made so far please try run it

#Task 2 part (II)
def pmf_plot(df4):
    for column in df4.columns:
        # Check if the column is categorical or a low-variance/low-unique numerical column
        if df4.dtypes[column] == 'object' or (
                df4.dtypes[column] in ['int64', 'float64'] and
                df4[column].nunique() < 10):
            plt.figure(figsize=(10, 5))

            # Calculate and plot PMF using value_counts(normalize=True)
            pmf = df4[column].value_counts(normalize=True)
            pmf.plot(kind='bar')

            plt.title(f"PMF of {column}")
            plt.xlabel(column)
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

#pmf_plot(df)

def plot_cond_pmf(df5):
    for column in df5.columns:
        if column == 'class':
            continue # TO avoid checking for class field
        condition = df5['class'].unique()

        for attack in condition:
            df_5_conditioned = df5[df5['class'] == attack]

            if df.dtypes[column] == 'object' or (
                df.dtypes[column] in ['int64', 'float64'] and
                df[column].nunique() < 10):
                plt.figure(figsize=(10,5))
                df5[column].value_counts(normalize=True).plot(color='blue', kind='bar', label='Original PMF')
                df_5_conditioned[column].value_counts(normalize=True).plot(color= 'orange', kind='bar', label= f'Conditional for {attack}')
                plt.title(f"PMF of {column} (Original and Conditional for {attack})")
                plt.legend()
                plt.grid(True)
                plt.show()

#plot_cond_pmf(df)

# Task 2 part (III)
# Summarize best-fit distributions for numerical columns
def document_best_fit_pdf(df61, class_column='class'):
    distributions = [
        stats.alpha, stats.norm, stats.expon, stats.gamma, stats.pareto, stats.beta, stats.lognorm, stats.weibull_min,
        stats.weibull_max, stats.t, stats.f, stats.chi2, stats.gumbel_r, stats.gumbel_l, stats.dweibull,
        stats.genextreme, stats.uniform, stats.arcsine, stats.cosine, stats.exponnorm, stats.foldcauchy
    ]
    result_summary = {}

    for column in df61.select_dtypes(include=[np.number]).columns:
        if column == class_column or df61[column].nunique() < 10 or df61[column].var() < 1e-5:
            print(f"Skipping '{column}' due to low variance or insufficient unique values.")
            continue

        # Filter extreme values (2nd to 98th percentiles)
        lower_bound = np.percentile(df61[column].dropna(), 0)
        upper_bound = np.percentile(df61[column].dropna(), 97)
        filtered_data = df61[(df61[column] >= lower_bound) & (df61[column] <= upper_bound)][column].dropna()

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
def document_pmf_data(df62, class_column='class'):
    pmf_summary = {}
    for column in df62.select_dtypes(include=['object']).columns:
        try:
            pmf_summary[column] = {}
            # Overall PMF
            overall_pmf = df62[column].value_counts(normalize=True).to_dict()
            pmf_summary[column]['overall'] = overall_pmf

            # Class-conditioned PMFs
            for class_value in df62[class_column].unique():
                conditioned_pmf = df62[df62[class_column] == class_value][column].value_counts(normalize=True).to_dict()
                pmf_summary[column][class_value] = conditioned_pmf
        except Exception as e:
            print(f"Error calculating PMF for '{column}': {e}")
    return pmf_summary

# Document results for both numerical and categorical columns
def document_analysis_results(df63):
    # Summarize best-fit distributions for numerical columns
    numerical_summary = document_best_fit_pdf(df63)
    # Summarize PMF data for categorical columns
    categorical_summary = document_pmf_data(df63)
    return numerical_summary, categorical_summary


def print_summary(numerical_summary, categorical_summary):
    """Print the summary of PDF and PMF"""
    print("\nPDF Summary (Numerical Columns):")
    for column, info in numerical_summary.items():
        print(f"  - Column: {column}")
        print(f"    Best-Fit Distribution: {info['best_fit_distribution']}") # Shape
        print(f"    Parameters: {info['params']}")  # First one is location parameter (Represents the central tendency or “location” of the distribution on the x-axis)
        #                                             and the second one is Scale parameter (Determines the spread or dispersion of the distribution)
        print(f"    Mean Squared Error (MSE): {info['mse']:.5f}")
        print("\n")

    print("PMF Summary (Categorical Columns):")
    for column, pmf in categorical_summary.items():
        print(f"\n  - Column: {column}") #Probabilities of each unique value in the column across the entire dataset
        print("    Overall PMF:")
        for value, probability in pmf['overall'].items():
            print(f"      {value}: {probability:.4f}")

        for class_value, class_pmf in pmf.items():
            if class_value == 'overall':
                continue
            print(f"    PMF Conditioned on {class_value}:") # Probabilities of each unique value given a specific class or condition.
            for value, probability in class_pmf.items():
                print(f"      {value}: {probability:.4f}")
    print("\n")
ns, cs = document_analysis_results(df)
print_summary(ns, cs)

"""Normal Distribution (norm)

	•	Parameters: (loc, scale)
	•	loc: Mean of the distribution.
	•	scale: Standard deviation.
	
	Gamma Distribution (gamma)

	•	Parameters: (shape, loc, scale)
	•	shape (a): Determines the skewness and tail behavior.
	•	loc: Location parameter (often 0).
	•	scale: Controls the spread.
	
	Exponential Distribution (expon)

	•	Parameters: (loc, scale)
	•	loc: Starting point of the distribution (minimum value).
	•	scale: 1/λ, where λ is the rate of decay.
	
	Lognormal Distribution (lognorm)

	•	Parameters: (shape, loc, scale)
	•	shape (s): Determines skewness.
	•	loc: Location parameter (shift on the x-axis).
	•	scale: Exp(mean of the underlying normal distribution).
	
	Weibull Distribution (weibull_min or weibull_max)

	•	Parameters: (shape, loc, scale)
	•	shape (c): Determines the behavior of the tail.
	•	loc: Location parameter.
	•	scale: Characteristic life or spread.
	
	Beta Distribution (beta)

	•	Parameters: (shape1, shape2, loc, scale)
	•	shape1 (α): Controls the left tail.
	•	shape2 (β): Controls the right tail.
	•	loc: Lower bound of the distribution.
	•	scale: Range of the distribution (difference between upper and lower bounds)."""