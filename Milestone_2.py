import numpy as np
import pandas as pd
from distfit import distfit
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

#import the dataframe
df = pd.read_csv("Train_data.csv")
#choose all the dataframe except the class column
selected_df = df.iloc[:,0:41]
#selected_df = df.drop(columns=['class'])
#display(selected_df.to_string())

training_df = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df = selected_df.iloc[int(df.shape[0]*0.7):, :]

#task 2 part 1 (i)
#to determine the best distribution we will use distfit as it trys the data on 89 different distributions
def best_fit_distribution_1(df1):
    dist = distfit()
    for column in df1.columns:
        if df1.dtypes[column] in ['int64', 'float64']:
            print(f"\nFitting distribution for column: {column}")
            try:
                # Filter data to exclude outliers (2nd to 98th percentiles)
                lower_bound = np.percentile(df1[column].dropna(), 2)
                upper_bound = np.percentile(df1[column].dropna(), 98)
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


def z_score(df2, thresholds1):
    # Get the weights (correlations) for numeric columns
    weights = attack_correlation(df2)
    print(f"Weights: {weights}")  # Debugging: Print weights

    result = {}
    predictions = []

    # Get the numeric columns
    numeric_columns = df2.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numeric Columns: {list(numeric_columns)}")  # Debugging: Print numeric columns

    # Filter out columns with zero correlation
    valid_columns = [col for col in numeric_columns if weights.get(col, 0) != 0]
    print(f"Valid Columns (Non-zero Correlation): {valid_columns}")  # Debugging: Print valid columns

    if not valid_columns:
        raise ValueError("No columns have a valid (non-zero) correlation")

    # Create a mapping of valid column names to their corresponding weights
    column_weights = {col: weights[col] for col in valid_columns}
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
                total_weighted_z_score += (weighted_z_score)

                # Store results by column and threshold for analysis
                if column not in result:
                    result[column] = {}

                for threshold in thresholds1:
                    # Count anomalies based on threshold for normal vs. anomaly separation
                    is_anomaly = weighted_z_score > threshold
                    result[column].setdefault(threshold, {'normal_count_above_threshold': 0, 'anomaly_count_above_threshold': 0})

                    # Track if the row is normal or anomaly for this threshold
                    if row['class'] == 'normal' and is_anomaly:
                        result[column][threshold]['normal_count_above_threshold'] += 1
                    elif row['class'] == 'anomaly' and is_anomaly:
                        result[column][threshold]['anomaly_count_above_threshold'] += 1

        # Determine the prediction based on total weighted Z-score
        anomaly_threshold = max(thresholds1)  # Adjust this logic as needed
        if total_weighted_z_score > anomaly_threshold:
            predictions.append(1)  # The 1 here is indicating anomaly
        else:
            predictions.append(0)  # The 0 here is indicating normal

    return predictions, result

# Example usage
thresholds = [1.5, 2.0, 2.5, 3.0]
predict, z_results = z_score(df, thresholds)
#print(predict)

def performance_metrics(df3, pridect_3):
    actual= (df3['class'] == 'anomaly').astype(int) # if the class data is an anomaly it will be stored as 1 and if the data
                                                    # is normal it will be stored as 0
    matrix = confusion_matrix(actual,pridect_3)
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

performance_metrics(df, predict)

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

#ns,cs = document_analysis_results(df)
#print_summary(ns, cs)
#best_fit_distribution_1(df)

#task 2 part(ii)
#I will calculate the conditioned pdf using the density thing in the histoplot
def plot_conditional_pdfs(df):
    # Initialize the distfit object
    dist = distfit()

    # Select only numerical columns, excluding the 'class' column
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if col != 'class']

    # Get unique values in the 'class' column to condition on
    class_values = df['class'].unique()

    # Define thresholds
    variance_threshold = 1e-3
    unique_values_threshold = 20

    # Loop through each numerical column and fit best distributions conditioned on each class value
    for column in numerical_columns:
        # Check for low variance or low unique values
        column_variance = df[column].var()
        column_unique_values = df[column].nunique()

        if column_variance < variance_threshold:
            print(f"Skipping '{column}' due to low variance.")
            continue  # Skip this column if variance is too low

        if column_unique_values < unique_values_threshold:
            print(f"Skipping '{column}' due to low unique values.")
            continue  # Skip this column if it has too few unique values

        # Plot PDFs conditioned on each class value
        plt.figure(figsize=(12, 6))
        plt.title(f'Best-Fitting PDFs of {column} Conditioned on Class Values')

        # Loop through each class value and fit the best PDF for that subset
        for value in class_values:
            # Filter the data for the current class value
            data_conditioned = df[df['class'] == value][column].dropna()

            if data_conditioned.var() < variance_threshold:
                print(f"Skipping '{column}' for class '{value}' due to low variance in subset.")
                continue

            # Fit the best distribution using distfit
            try:
                dist.fit_transform(data_conditioned ,  verbose=0)
                # Plot the best fit
                dist.plot()
                plt.plot([], [], ' ', label=f'class = {value}')
            except Exception as e:
                print(f"Error fitting distribution for column '{column}' with class '{value}': {e}")
                continue  # Skip if fitting fails for this subset

        # Display the plot with all class-based PDFs for the column
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend(title="Class")
        plt.grid(True)
        plt.show()


# Call the function with the DataFrame
#plot_conditional_pdfs(df)

# Function to calculate MSE between empirical and fitted PDF
def calculate_mse(empirical_pdf, fitted_pdf):
    return np.mean((empirical_pdf - fitted_pdf) ** 2)

# Function to fit distributions and evaluate the best fit based on MSE
# Function to fit distributions and evaluate the best fit based on MSE
def best_fit_distribution(data, distributions, adjust_range=False):
    best_distribution = None
    best_mse = float('inf')
    best_params = None

    # Determine calculation range based on data characteristics
    if adjust_range:
        lower_bound, upper_bound = np.percentile(data, [2, 98])  # Adjusted range for columns with extreme values
    else:
        lower_bound, upper_bound = data.min(), data.max()  # Full range for stable columns

    bin_edges = np.linspace(lower_bound, upper_bound, 30)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Create empirical PDF for the selected range
    counts, _ = np.histogram(data, bins=bin_edges, density=True)

    # Test each distribution
    for distribution in distributions:
        try:
            # Fit the distribution to the data
            params = distribution.fit(data)

            # Calculate the PDF with the fitted parameters
            fitted_pdf = distribution.pdf(bin_centers, *params)

            # Calculate MSE for the fit within the adjusted range
            mse = calculate_mse(counts, fitted_pdf)
            #print(f"mse for {distribution} is {mse}")

            # Identify the distribution with the lowest MSE
            if mse < best_mse:
                best_distribution = distribution
                best_mse = mse
                best_params = params
        except (ValueError, OverflowError, RuntimeError) as e:
            print(f"Error fitting {distribution.name}: {e}")
            continue

    return best_distribution, best_params, best_mse


# Function to determine if the column has extreme values
def has_extreme_values(data):
    # Calculate IQR and range
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    data_range = np.max(data) - np.min(data)

    # Thresholds to detect high variability or extreme values
    if iqr > 1.5 * data_range or data_range > 10 * iqr:
        return True  # Indicates extreme values are likely present
    return False


def best_fit_mse(df):
    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Get unique values in the 'class' column to condition on
    class_values = df['class'].unique()

    # Comprehensive list of candidate distributions to test from scipy.stats
    distributions = [
        stats.norm, stats.expon, stats.gamma, stats.pareto, stats.beta, stats.lognorm, stats.weibull_min,
        stats.weibull_max, stats.t, stats.f, stats.chi2, stats.gumbel_r, stats.gumbel_l, stats.dweibull,
        stats.genextreme, stats.uniform
    ]

    # Loop through each numerical column and fit the best distributions conditioned on each class value
    for column_name in numerical_columns:
        column = df[column_name]  # Reference the actual column data here

        # Check if the column has extreme values
        adjust_range = has_extreme_values(column)
        plt.figure(figsize=(12, 6))
        plt.title(f'Best-Fitting PDFs of {column_name} Conditioned on Class Values (Based on MSE)')

        if column.nunique() < 10:
            print(f"Skipping '{column_name}' due to insufficient unique values after filtering.")
            continue  # Skip this column if there are too few unique values

        if column.var() < 1e-5:
            print(f"Skipping '{column_name}' due to low variance.")
            continue  # Skip this column if variance is too low

        # Loop through each class value and fit the best PDF for that subset
        for value in class_values:
            # Filter the data for the current class value
            data_conditioned = df[df['class'] == value][column_name].dropna()
            # Find the best-fitting distribution based on MSE
            best_distribution, best_params, best_mse = best_fit_distribution(data_conditioned, distributions,
                                                                             adjust_range=adjust_range)

            # Plot the best fit along with the empirical data
            counts, bin_edges, _ = plt.hist(data_conditioned, bins=30, density=True, alpha=0.5,
                                            label=f'class = {value} (Empirical)')
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

            # Plot the fitted PDF over the selected range
            if best_distribution is not None:
                fitted_pdf = best_distribution.pdf(bin_centers, *best_params)
                plt.plot(bin_centers, fitted_pdf,
                         label=f'{best_distribution.name} (MSE = {best_mse:.5f}, class = {value})')

        # Display plot settings
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
#best_fit_mse(df)

def pmf_plot(df):
    for column in df.columns:
        # Check if the column is categorical or a low-variance/low-unique numerical column
        if df.dtypes[column] == 'object' or (
                df.dtypes[column] in ['int64', 'float64'] and
                (df[column].var() < 1e-5 or df[column].nunique() < 10)
        ):
            plt.figure(figsize=(10, 5))

            # Calculate and plot PMF using value_counts(normalize=True)
            pmf = df[column].value_counts(normalize=True)
            pmf.plot(kind='bar')

            plt.title(f"PMF of {column}")
            plt.xlabel(column)
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

#pmf_plot(df)

def plot_cond_pmf(df_5):
    for column in df_5.columns:
        if column == 'class':
            continue # TO avoid checking for class field
        condition = df_5['class'].unique()

        for attack in condition:
            df_5_conditioned = df_5[df_5['class'] == attack]

            if df.dtypes[column] == 'object' or (
                df.dtypes[column] in ['int64', 'float64'] and
                (df[column].var() < 1e-5 or df[column].nunique() < 10)
            ):
                plt.figure(figsize=(10,5))
                df_5[column].value_counts(normalize=True).plot(color= 'blue',kind='bar', label= 'Original PMF')
                df_5_conditioned[column].value_counts(normalize=True).plot(color= 'orange', kind='bar', label= f'Conditional for {attack}')
                plt.title(f"PMF of {column} (Original and Conditional for {attack})")
                plt.legend()
                plt.grid(True)
                plt.show()

#plot_cond_pmf(df)
#best_fit_distribution(df)
