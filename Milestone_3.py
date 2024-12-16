import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from Milestone_2 import document_analysis_results, document_best_fit_pdf, document_pmf_data
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
    anomaly_conditioned_data = {}
    normal_conditioned_data = {}
    for column in df_1:
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

numerical_part_best_fits_anomaly, categorical_part_best_fits_anomaly = document_analysis_results_ms3(anomaly_conditioned)
numerical_part_best_fit_normal, categorical_part_best_fit_normal = document_analysis_results_ms3(normal_conditioned)
numerical_part_best_fit_nocond, categorical_part_best_fit_nocond = document_analysis_results_ms3(training_df_no_class)

# Extracting best fits for each column as a dictionary
numerical_best_fit_anomaly = {
    'anomaly': {col: numerical_part_best_fits_anomaly[col]['best_fit'] for col in numerical_part_best_fits_anomaly}
}

categorical_best_fit_anomaly = {
    'anomaly': {col: categorical_part_best_fits_anomaly[col]['best_fit'] for col in categorical_part_best_fits_anomaly}
}

numerical_best_fit_normal = {
    'normal': {col: numerical_part_best_fit_normal[col]['best_fit'] for col in numerical_part_best_fit_normal}
}

categorical_best_fit_normal = {
    'normal': {col: categorical_part_best_fit_normal[col]['best_fit'] for col in categorical_part_best_fit_normal}
}

# Extracting best fits for non-conditioned data
numerical_best_fit_nocond = {
    col: numerical_part_best_fit_nocond[col]['best_fit'] for col in numerical_part_best_fit_nocond
}

categorical_best_fit_nocond = {
    col: categorical_part_best_fit_nocond[col]['best_fit'] for col in categorical_part_best_fit_nocond
}

def calculate_pdf_or_pmf(values, best_fit_params, is_categorical=False):
    """
    Calculate the PDF (numerical) or PMF (categorical) values based on the best-fit parameters.
    """
    if is_categorical:
        # For categorical data, use a lookup and handle unseen categories
        return values.map(lambda x: best_fit_params.get(x, 1e-6))  # Small probability for unseen categories
    else:
        # For numerical data, best_fit_params is assumed to be a distribution object with a .pdf() method
        return best_fit_params.pdf(values)

def naiive_bayes(df_res):

    # Separate numerical and categorical columns
    numerical_cols = list(numerical_best_fit_anomaly.keys())
    categorical_cols = list(categorical_best_fit_anomaly.keys())

    # Compute numerator (conditioned on 'Anomaly')
    num_numerator_anomaly = np.prod(
        [calculate_pdf_or_pmf(df_res[col], numerical_best_fit_anomaly[col]) for col in numerical_cols], axis=0
    )
    cat_numerator_anomaly = np.prod(
        [calculate_pdf_or_pmf(df_res[col], categorical_best_fit_anomaly[col], is_categorical=True) for col in categorical_cols], axis=0
    )
    numerator_anomaly = num_numerator_anomaly * cat_numerator_anomaly

    # Compute numerator (conditioned on 'Anomaly')
    num_numerator_normal = np.prod(
        [calculate_pdf_or_pmf(df_res[col], numerical_best_fit_normal[col]) for col in numerical_cols], axis=0
    )
    cat_numerator_normal = np.prod(
        [calculate_pdf_or_pmf(df_res[col], categorical_best_fit_normal[col], is_categorical=True) for col in categorical_cols], axis=0
    )
    numerator_normal = num_numerator_normal * cat_numerator_normal

    # Compute denominator (no conditioning)
    num_denominator = np.prod(
        [calculate_pdf_or_pmf(df_res[col], numerical_best_fit_nocond[col]) for col in numerical_cols], axis=0
    )
    cat_denominator = np.prod(
        [calculate_pdf_or_pmf(df_res[col], categorical_best_fit_nocond[col], is_categorical=True) for col in categorical_cols], axis=0
    )
    denominator = num_denominator * cat_denominator

    pr_normal_given_row = numerator_normal / denominator
    pr_anomaly_given_row = numerator_anomaly / denominator
    predicts = np.where(pr_anomaly_given_row > pr_normal_given_row, 'anomaly', 'normal')
    return predicts

training_predict = naiive_bayes(training_df_no_class)
predictions = naiive_bayes(testing_df_no_class)


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

performance_metrics(training_attack, training_predict)

#performance_metrics(testing_attack, predictions)


