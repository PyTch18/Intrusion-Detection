import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve
import warnings
from Milestone_2 import attack_correlation

# Removing some warnings appearing due to the large dataset
# No effects on the output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Function: Encode Categorical Features
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


# Function: Apply PCA with Non-Negative Shift for MultinomialNB
def apply_pca(train_df, test_df, n_components=10):
    """
    Reduces dimensionality using PCA after scaling the data.
    Shifts PCA output to ensure non-negative values for MultinomialNB.
    """
    # Separate features and target variable
    y_train = train_df['class']
    X_train = train_df.drop(columns=['class'])
    y_test = test_df['class']
    X_test = test_df.drop(columns=['class'])

    # Scale the features before applying PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Ensure non-negative values for MultinomialNB
    X_train_pca_nonneg = X_train_pca - X_train_pca.min(axis=0)
    X_test_pca_nonneg = X_test_pca - X_test_pca.min(axis=0)

    # Convert back to DataFrame
    train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
    train_pca_df_nonneg = pd.DataFrame(X_train_pca_nonneg, columns=[f'PC{i + 1}' for i in range(n_components)])

    test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
    test_pca_df_nonneg = pd.DataFrame(X_test_pca_nonneg, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Add the target variable back
    train_pca_df['class'] = y_train.reset_index(drop=True)
    test_pca_df['class'] = y_test.reset_index(drop=True)

    train_pca_df_nonneg['class'] = y_train.reset_index(drop=True)
    test_pca_df_nonneg['class'] = y_test.reset_index(drop=True)

    return train_pca_df, test_pca_df, train_pca_df_nonneg, test_pca_df_nonneg


# Function: Binarize Data for BernoulliNB
def binarize_data(train_df, test_df, threshold=0.0):
    """
    Binarizes the numeric data for BernoulliNB.
    """
    binarizer = Binarizer(threshold=threshold)
    X_train_bin = binarizer.fit_transform(train_df.drop(columns=['class']))
    X_test_bin = binarizer.transform(test_df.drop(columns=['class']))

    # Convert to DataFrame and add class column
    train_bin_df = pd.DataFrame(X_train_bin, columns=train_df.columns[:-1])
    train_bin_df['class'] = train_df['class'].values

    test_bin_df = pd.DataFrame(X_test_bin, columns=test_df.columns[:-1])
    test_bin_df['class'] = test_df['class'].values

    return train_bin_df, test_bin_df


# Function: Train and Evaluate Models
def train_and_evaluate_models(train_df, test_df, train_bin_df, test_bin_df, train_df_nonneg, test_df_nonneg):
    """
    Trains GaussianNB, MultinomialNB, and BernoulliNB models and evaluates their performance.
    """
    # Prepare data
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    X_train_nonneg = train_df_nonneg.drop(columns=['class'])
    y_train_nonneg = train_df_nonneg['class']
    X_test_nonneg = test_df_nonneg.drop(columns=['class'])
    y_test_nonneg = test_df_nonneg['class']

    # BernoulliNB requires binary input
    X_train_bin = train_bin_df.drop(columns=['class'])
    y_train_bin = train_bin_df['class']
    X_test_bin = test_bin_df.drop(columns=['class'])
    y_test_bin = test_bin_df['class']

    # Initialize models
    gaussian_nb = GaussianNB()
    multinomial_nb = MultinomialNB()
    bernoulli_nb = BernoulliNB()

    # Train and evaluate GaussianNB
    print("\n--- Gaussian Naïve Bayes Performance ---")
    gaussian_nb.fit(X_train, y_train)
    y_pred_gnb = gaussian_nb.predict(X_test)
    print_metrics(y_test, y_pred_gnb)

    # Train and evaluate MultinomialNB
    print("\n--- Multinomial Naïve Bayes Performance ---")
    multinomial_nb.fit(X_train_nonneg, y_train_nonneg)
    y_pred_mnb = multinomial_nb.predict(X_test_nonneg)
    print_metrics(y_test_nonneg, y_pred_mnb)

    # Train and evaluate BernoulliNB
    print("\n--- Bernoulli Naïve Bayes Performance ---")
    bernoulli_nb.fit(X_train_bin, y_train_bin)
    y_pred_bnb = bernoulli_nb.predict(X_test_bin)
    print_metrics(y_test_bin, y_pred_bnb)


# Function: Print Metrics
def print_metrics(y_true, y_pred):
    """
    Prints accuracy, precision, recall, and confusion matrix.
    """
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='anomaly'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, pos_label='anomaly'):.4f}")


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
            fitted_pdf = distribution.pdf(bin_centers, *params)

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


def document_best_fit_pdf_pca(df61, class_column='class'):
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
        lower_bound = np.percentile(df61[column].dropna(), 2)
        upper_bound = np.percentile(df61[column].dropna(), 98)
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


def document_analysis_results_ms3_pca(dict_1):
    # Transform dictionaries into DataFrames
    df_1_pca = pd.DataFrame(dict_1)

    # Pass DataFrames to the document functions
    numerical_summary = document_best_fit_pdf_pca(df_1_pca)

    return numerical_summary


def conditioned_data_pca(df_1_pca):
    condition = df_1_pca['class'].unique()
    df_1_no_class = df_1_pca.copy()
    df_1_no_class = df_1_no_class.drop('class', axis=1)
    anomaly_conditioned_data = {}
    normal_conditioned_data = {}
    for column in df_1_no_class.columns:
        for value in condition:
            if value == 'anomaly':
                # Collecting the anomaly conditioned values together
                anomaly_conditioned_data[column] = df_1_pca[df_1_pca['class'] == value][column].dropna()
            else:
                # Collecting the normal conditioned data together
                normal_conditioned_data[column] = df_1_pca[df_1_pca['class'] == value][column].dropna()
    return anomaly_conditioned_data, normal_conditioned_data


def calculate_pdf(values, best_fit_params):
    """
    Calculate the PDF (numerical) values based on the best-fit parameters.

    Parameters:
    - values: The data values for which PDF/PMF is to be calculated.
    - best_fit_params: Dictionary containing 'distribution' and 'params' for numerical.
    """

    try:
        distribution_name = best_fit_params.get('best_fit_distribution')  # Get the distribution name
        params = best_fit_params.get('params', {})  # Get the parameters (dict)

        if not distribution_name:
            raise ValueError("Missing 'best_fit_distribution' key in best_fit_params.")

        # Dynamically fetch the distribution object from scipy.stats
        distribution = getattr(stats, distribution_name)

        if isinstance(params, tuple):
            params = list(params)  # Convert tuple to list
        print(params)
        # Calculate PDF values with the extracted parameters
        print(values)

        log_pdf = np.log(distribution.pdf(values, *params) + 1e-10) # To avoid the nan and inf problems

        return log_pdf  # Pass params as a mapping

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

    # Function to safely calculate PDF, skipping invalid columns
    def safe_calculate(col, fit_params):
        try:
            if fit_params and 'best_fit_distribution' in fit_params and fit_params['best_fit_distribution']:
                print("data fitted and getting values")
                return calculate_pdf(df_res[col], fit_params)
            else:
                # If parameters are invalid, skip by returning 1
                print("data not fitted")
                return np.ones(len(df_res[col]))
        except Exception as e:
            # Catch any unexpected errors and return 1 to avoid breaking the product
            print(f"Skipping column '{col}' due to error: {e}")
            return np.ones(len(df_res[col]))

    # Obtain weights
    weights = attack_correlation(train_df_pca)
    #print('\nWeights:')
    #print(weights)
    #print('\n')

    # --- Compute numerator conditioned on 'Anomaly' ---
    numerical_cols_anomaly = list(numerical_part_best_fit_anomaly.keys())
    numerator_anomaly = np.prod([
        safe_calculate(col, numerical_part_best_fit_anomaly.get(col)) * weights.get(col, 1)
        for col in numerical_cols_anomaly
    ], axis=0)
    #print("\nNumerator Anomaly:")
    #print(numerator_anomaly)

    # --- Compute numerator conditioned on 'Normal' ---
    numerical_cols_normal = list(numerical_part_best_fit_normal.keys())
    numerator_normal = np.prod([
        safe_calculate(col, numerical_part_best_fit_normal.get(col)) * weights.get(col, 1)
        for col in numerical_cols_normal
    ], axis=0)
    #print("\nNumerator Normal:")
    #print(numerator_normal)

    # --- Compute denominator without conditioning ---
    numerical_cols_nocond = list(numerical_part_best_fit_nocond.keys())
    denominator = np.prod([
        safe_calculate(col, numerical_part_best_fit_nocond.get(col)) * weights.get(col, 1)
        for col in numerical_cols_nocond
    ], axis=0)
    #print("\nDenominator:")
    #print(denominator)

    # --- Calculate posterior probabilities ---
    pr_normal_given_row = numerator_normal / denominator
    pr_anomaly_given_row = numerator_anomaly / denominator

    #print("\nPosterior Probability (Normal):")
    #print(pr_normal_given_row)
    #print("\nPosterior Probability (Anomaly):")
    #print(pr_anomaly_given_row)


    predicts = np.where(
        (abs(pr_anomaly_given_row)*60 > abs(pr_normal_given_row)),
        'anomaly',
        'normal'
    )

    # Convert predictions to 0 and 1
    predictions_final = [1 if i == 'anomaly' else 0 for i in predicts]

    return predictions_final


def performance_metrics(attack_3, predict_3):
    attack_3 = attack_3.apply(lambda x: 1 if x == 'anomaly' else 0)
    matrix = confusion_matrix(attack_3, predict_3)
    if matrix.shape == (2, 2):
        """True Negative (tn): Correctly predicted normal points.
           False Positive (fp): Normal points wrongly predicted as anomalies.
           False Negative (fn): Anomalies wrongly predicted as normal.
           True Positive (tp): Correctly predicted anomalies."""
        tn, fp, fn, tp = matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
    else:
        print("wrong data set")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load dataset
    file_path = "Train_data.csv"
    df = pd.read_csv(file_path)


    New_testing = pd.read_csv('Test_data.csv')


    # Split dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # Encode categorical features
    #train_df, test_df = encode_categorical_features(train_df, test_df)
    train_df, New_testing = encode_categorical_features(train_df, New_testing)


    train_df_pca, New_testing_pca, train_df_pca_nonneg, New_testing_nonneg = apply_pca(train_df, New_testing, n_components=10)

    # Apply PCA for dimensionality reduction
    #train_df_pca, test_df_pca, train_df_pca_nonneg, test_df_pca_nonneg = apply_pca(train_df, test_df, n_components=10)

    # Binarize data for BernoulliNB
    #train_bin_df, test_bin_df = binarize_data(train_df_pca, test_df_pca, threshold=0.0)
    train_bin_df, New_testing_bin_df = binarize_data(train_df_pca, New_testing_pca, threshold=0.0)


    # Train and evaluate all models
    #train_and_evaluate_models(train_df_pca, test_df_pca, train_bin_df, test_bin_df, train_df_pca_nonneg, test_df_pca_nonneg)
    train_and_evaluate_models(train_df_pca, New_testing_pca, train_bin_df, New_testing_bin_df, train_df_pca_nonneg, New_testing_nonneg)


    # Applying PCA for Task 1

    # Note all the columns are numerical after applying PCA
    anomaly_conditioned, normal_conditioned = conditioned_data_pca(train_df_pca)

    print("\nFitting the best-fit distributions for the PCA data... \n")
    numerical_part_best_fit_anomaly = document_analysis_results_ms3_pca(anomaly_conditioned)
    numerical_part_best_fit_normal= document_analysis_results_ms3_pca(normal_conditioned)
    train_df_pca_attack = train_df_pca_nonneg['class']
    train_df_pca_no_class = train_df_pca_nonneg.drop(columns=['class'])
    numerical_part_best_fit_nocond= document_analysis_results_ms3_pca(train_df_pca_no_class)

    #print("\nDebug\n")
    #print(numerical_part_best_fit_nocond)
    #print("\n")

    #test_df_pca_attack = test_df_pca['class']
    New_testing_df_pca_attack = New_testing_nonneg['class']

    #test_df_pca_no_class = test_df_pca.drop(columns=['class'])
    New_testing_df_pca_no_class = New_testing_nonneg.drop(columns=['class'])

    #print(numerical_part_best_fit_anomaly)
    #print(numerical_part_best_fit_normal)
    #print(numerical_part_best_fit_nocond)
    print("\nPCA data fitted.\n")
    # training_predict = pd.Series(naiive_bayes(train_df_pca_no_class))

    print("\nCalculating Naive Bayes predictions...\n")
    # predictions = pd.Series(naiive_bayes(test_df_pca_no_class))
    predictions_new = pd.Series(naiive_bayes(New_testing_df_pca_no_class))
    # performance_metrics(train_df_pca_attack, training_predict)

    # performance_metrics(test_df_pca_attack, predictions)
    performance_metrics(New_testing_df_pca_attack, predictions_new)
