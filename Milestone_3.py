import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from Milestone_2 import document_analysis_results, document_best_fit_pdf, document_pmf_data, performance_metrics
from Task_1 import categorical

#import the dataframe
df = pd.read_csv("Train_data.csv")

training_df = df.iloc[:int(df.shape[0]*0.7),:]
training_attack = training_df['class']
testing_df = df.iloc[int(df.shape[0]*0.7):, :]
testing_attack = testing_df['class']

selected_df = df.iloc[:,0:41] #Selecting the columns without the class column
training_df_no_class = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df_no_class = selected_df.iloc[int(df.shape[0]*0.7):, :]

def document_analysis_results_ms3(dict_1):

    # Transform dictionaries into DataFrames
    df_1 = pd.DataFrame(dict_1)
    
    # Pass DataFrames to the document functions
    numerical_summary = document_best_fit_pdf(df_1)
    categorical_summary = document_pmf_data(df_1)

    return numerical_summary, categorical_summary


def conditioned_data(df_1):
    condition = df_1['class'].unique()
    df_1_no_class = df_1.copy()
    df_1_no_class = df_1_no_class.drop('class', axis=1)
    anomaly_conditioned_data = {}
    normal_conditioned_data = {}
    for column in df_1_no_class.columns:
        for value in condition:
            if value == 'anomaly':
                #Collecting the anomaly conditioned values together
                anomaly_conditioned_data[column] = df_1[df_1['class'] == value][column].dropna()
            else:
                # Collecting the normal conditioned data together
                normal_conditioned_data[column] = df_1[df_1['class'] == value][column].dropna()
    return anomaly_conditioned_data, normal_conditioned_data

# Preparing the data for best fitting by storing in two different sets
anomaly_conditioned, normal_conditioned= conditioned_data(training_df)

numerical_part_best_fit_anomaly, categorical_part_best_fit_anomaly = document_analysis_results_ms3(anomaly_conditioned)
numerical_part_best_fit_normal, categorical_part_best_fit_normal = document_analysis_results_ms3(normal_conditioned)
numerical_part_best_fit_nocond, categorical_part_best_fit_nocond = document_analysis_results_ms3(training_df_no_class)


'''
# Extracting best fits for each column as a dictionary (anomaly conditioned)
numerical_best_fit_anomaly = {
    col: numerical_part_best_fit_anomaly[col]['best_fit_distribution']
    for col in numerical_part_best_fit_anomaly if 'best_fit_distribution' in numerical_part_best_fit_anomaly[col]
}

categorical_best_fit_anomaly = {
    col: categorical_part_best_fit_anomaly[col]['best_fit_distribution']
    for col in categorical_part_best_fit_anomaly if 'best_fit_distribution' in categorical_part_best_fit_anomaly[col]
}

# Corrected dictionaries for normal-conditioned values
numerical_best_fit_normal = {
    col: numerical_part_best_fit_normal[col]['best_fit_distribution']
    for col in numerical_part_best_fit_normal if 'best_fit_distribution' in numerical_part_best_fit_normal[col]
}

categorical_best_fit_normal = {
    col: categorical_part_best_fit_normal[col]['best_fit_distribution']
    for col in categorical_part_best_fit_normal if 'best_fit_distribution' in categorical_part_best_fit_normal[col]
}

# Extracting best fits for non-conditioned data
numerical_best_fit_nocond = {
    col: numerical_part_best_fit_nocond[col]['best_fit_distribution']
    for col in numerical_part_best_fit_nocond if 'best_fit_distribution' in numerical_part_best_fit_nocond[col]
}

categorical_best_fit_nocond = {
    col: categorical_part_best_fit_nocond[col]['best_fit_distribution']
    for col in categorical_part_best_fit_nocond if 'best_fit_distribution' in categorical_part_best_fit_nocond[col]
}
'''


def calculate_pdf_or_pmf(values, best_fit_params, is_categorical):
    """
    Calculate the PDF (numerical) or PMF (categorical) values based on the best-fit parameters.

    Parameters:
    - values: The data values for which PDF/PMF is to be calculated.
    - best_fit_params: Dictionary containing 'distribution' and 'params' for numerical,
                       or a mapping of probabilities for categorical data.
    - is_categorical: Boolean flag indicating whether the data is categorical.
    """
    if is_categorical:
        # For categorical data, map the probabilities
        return values.map(lambda x: best_fit_params.get(x, 1e-6))  # Small probability for unseen categories
    else:
        # For numerical data, extract the distribution and parameters
        try:
            distribution_name = best_fit_params.get('best_fit_distribution')  # Get the distribution name
            params = best_fit_params.get('params', {})  # Get the parameters (dict)

            if not distribution_name:
                raise ValueError("Missing 'distribution' key in best_fit_params.")

            # Dynamically fetch the distribution object from scipy.stats
            distribution = getattr(stats, distribution_name)

            # Calculate PDF values with the extracted parameters
            return distribution.pdf(values, **params)
        except Exception as e:
            print(f"Error calculating PDF: {e}")
            return None


def naiive_bayes(df_res):
    """
    Perform Naive Bayes prediction for anomaly detection.

    Parameters:
    - df_res: DataFrame containing the input data (excluding the class column).

    Returns:
    - List of predictions: 1 for 'anomaly', 0 for 'normal'.
    """

    # Function to safely calculate PDF/PMF, skipping invalid columns
    def safe_calculate(col, fit_params, is_categorical=False):
        
        try:
            if fit_params and 'distribution' in fit_params and fit_params['distribution']:
                return calculate_pdf_or_pmf(df_res[col], fit_params, is_categorical)
            elif is_categorical and isinstance(fit_params, dict):
                return calculate_pdf_or_pmf(df_res[col], fit_params, is_categorical=True)
            else:
                # If parameters are invalid, skip by returning 1
                return np.ones(len(df_res[col]))
        except Exception as e:
            # Catch any unexpected errors and return 1 to avoid breaking the product
            print(f"Skipping column '{col}' due to error: {e}")
            return np.ones(len(df_res[col]))

    # --- Compute numerator conditioned on 'Anomaly' ---
    numerical_cols = list(numerical_part_best_fit_anomaly.keys())
    categorical_cols = list(categorical_part_best_fit_anomaly.keys())

    num_numerator_anomaly = np.prod(
        [safe_calculate(col, numerical_part_best_fit_anomaly.get(col)) for col in numerical_cols], axis=0
    )
    cat_numerator_anomaly = np.prod(
        [safe_calculate(col, categorical_part_best_fit_anomaly.get(col), is_categorical=True) for col in
         categorical_cols], axis=0
    )
    numerator_anomaly = num_numerator_anomaly * cat_numerator_anomaly

    # --- Compute numerator conditioned on 'Normal' ---
    numerical_cols = list(numerical_part_best_fit_normal.keys())
    categorical_cols = list(categorical_part_best_fit_normal.keys())

    num_numerator_normal = np.prod(
        [safe_calculate(col, numerical_part_best_fit_normal.get(col)) for col in numerical_cols], axis=0
    )
    cat_numerator_normal = np.prod(
        [safe_calculate(col, categorical_part_best_fit_normal.get(col), is_categorical=True) for col in
         categorical_cols], axis=0
    )
    numerator_normal = num_numerator_normal * cat_numerator_normal

    # --- Compute denominator without conditioning ---
    numerical_cols = list(numerical_part_best_fit_nocond.keys())
    categorical_cols = list(categorical_part_best_fit_nocond.keys())

    num_denominator = np.prod(
        [safe_calculate(col, numerical_part_best_fit_nocond.get(col)) for col in numerical_cols], axis=0
    )
    cat_denominator = np.prod(
        [safe_calculate(col, categorical_part_best_fit_nocond.get(col), is_categorical=True) for col in
         categorical_cols], axis=0
    )
    denominator = num_denominator * cat_denominator

    # --- Calculate posterior probabilities ---
    pr_normal_given_row = numerator_normal / denominator
    pr_anomaly_given_row = numerator_anomaly / denominator

    # --- Generate predictions ---
    predicts = np.where(pr_anomaly_given_row > pr_normal_given_row, 'anomaly', 'normal')

    # Convert predictions to 0 and 1
    predictions_final = [1 if i == 'anomaly' else 0 for i in predicts]

    return predictions_final

training_predict = naiive_bayes(training_df_no_class)
#predictions = naiive_bayes(testing_df_no_class)


performance_metrics(training_attack, training_predict)

#performance_metrics(testing_attack, predictions)


