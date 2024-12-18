import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from Milestone_2 import document_best_fit_pdf, document_pmf_data, performance_metrics
import warnings

# Removing some warnings appearing due to the large dataset
# No effects on the output
warnings.filterwarnings("ignore", category=RuntimeWarning)

#import the dataframe
df = pd.read_csv("Train_data.csv")

training_df = df.iloc[:int(df.shape[0]*0.7),:]
training_attack = training_df['class']
testing_df = df.iloc[int(df.shape[0]*0.7):, :]
testing_attack = testing_df['class']

selected_df = df.iloc[:,0:41] #Selecting the columns without the class column
training_df_no_class = selected_df.iloc[:int(df.shape[0]*0.7),:]
testing_df_no_class = selected_df.iloc[int(df.shape[0]*0.7):, :]

#Task 1
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
        # Extract the nested dictionary if needed
        probabilities = best_fit_params.get('overall', best_fit_params)

        # Map the values to probabilities with a fallback for unseen categories
        mapped_probs = values.map(lambda x: probabilities.get(x, 1e-6))

        return mapped_probs
    else:
        # For numerical data, extract the distribution and parameters
        try:
            distribution_name = best_fit_params.get('best_fit_distribution')  # Get the distribution name
            params = best_fit_params.get('params', {})  # Get the parameters (dict)

            if not distribution_name:
                raise ValueError("Missing 'distribution' key in best_fit_params.")

            # Dynamically fetch the distribution object from scipy.stats
            distribution = getattr(stats, distribution_name)
            if isinstance(params, tuple):
                params = list(params)  # Convert tuple to list
            # Calculate PDF values with the extracted parameters

            log_pdf = np.log(distribution.pdf(values, *params) + 1e-10)  # To avoid the nan and inf problems

            return log_pdf  # Unpack params as positional arguments

        except Exception as e:
            print(f"Error calculating PDF: {e}")
            return None


def naiive_bayes(df_res):
    print("\nCalculating Naive Bayes predictions...\n")
    """
    Perform Naive Bayes prediction for anomaly detection.

    Parameters:
    - df_res: DataFrame containing the input data (excluding the class column).

    Returns:
    - List of predictions: 1 for 'anomaly', 0 for 'normal'.
    """

    def safe_calculate(col, fit_params, is_categorical=False):
        try:
            if fit_params and 'best_fit_distribution' in fit_params and fit_params['best_fit_distribution']:
                return calculate_pdf_or_pmf(df_res[col], fit_params, is_categorical)
            elif is_categorical and isinstance(fit_params, dict):
                return calculate_pdf_or_pmf(df_res[col], fit_params, is_categorical=True)
            else:
                # If parameters are invalid, return 1 to skip this column
                return np.ones(len(df_res[col]))
        except Exception as e:
            # Log any unexpected errors and return 1
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

    # Handle potential issues with invalid probabilities
    pr_normal_given_row = np.nan_to_num(pr_normal_given_row, nan=1e-10, posinf=1e10, neginf=1e-10)
    pr_anomaly_given_row = np.nan_to_num(pr_anomaly_given_row, nan=1e-10, posinf=1e10, neginf=1e-10)

    print("\nProbabilities")
    print(pr_normal_given_row)
    print("")
    print(pr_anomaly_given_row)
    print("\n")
    # --- Generate predictions ---

    predicts = np.where(pr_anomaly_given_row>=pr_normal_given_row, 'anomaly', 'normal')
    # Convert predictions to 0 and 1
    predictions_final = [1 if i == 'anomaly' else 0 for i in predicts]

    print(predictions_final)

    return predictions_final


#training_predict = pd.Series(naiive_bayes(training_df_no_class))
predictions = pd.Series(naiive_bayes(testing_df_no_class))
#New_testing = pd.read_csv('Test_data.csv')
#New_testing_attack = New_testing['class']
#New_testing_no_attack = New_testing.drop(columns=['class'])
#predictions_new = pd.Series(naiive_bayes(New_testing))

#performance_metrics(training_attack, training_predict)

performance_metrics(testing_attack, predictions)
#performance_metrics(New_testing_attack, predictions_new)
print("\n")

#Task 2
def encode_categorical_features(train_df_task_2, test_df_task_2):
    """
    Perform one-hot encoding for categorical features in train and test datasets using the same encoder.
    """
    # Select categorical columns
    categorical_columns = train_df_task_2.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'class']

    print("Encoding the following categorical columns:")
    for col in categorical_columns:
        print(f"- {col}")

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train_data = encoder.fit_transform(train_df_task_2[categorical_columns])
    encoded_test_data = encoder.transform(test_df_task_2[categorical_columns])

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    # Convert encoded data to DataFrames
    encoded_train_df = pd.DataFrame(encoded_train_data, columns=encoded_columns, index=train_df_task_2.index)
    encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoded_columns, index=test_df_task_2.index)

    # Drop original categorical columns and concatenate encoded ones
    train_df_task_2 = train_df_task_2.drop(columns=categorical_columns).join(encoded_train_df)
    test_df_task_2 = test_df_task_2.drop(columns=categorical_columns).join(encoded_test_df)

    return train_df_task_2, test_df_task_2


def train_and_evaluate_models(train_df, test_df):
    """
    Train and evaluate Gaussian, Multinomial, and Bernoulli Naïve Bayes models.
    """
    # Separate features and labels
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    # Initialize models
    models = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        predictions_task_2 = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions_task_2)
        precision = precision_score(y_test, predictions_task_2, pos_label='anomaly', zero_division=1)
        recall = recall_score(y_test, predictions_task_2, pos_label='anomaly', zero_division=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


# Split dataset
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# One-hot encode categorical features
train_df, test_df = encode_categorical_features(train_df, test_df)

# Train and evaluate models
train_and_evaluate_models(train_df, test_df)


