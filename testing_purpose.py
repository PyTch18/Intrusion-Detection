import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

def encode_categorical_features(train_df, test_df):
    """
    Perform one-hot encoding for categorical features in train and test datasets using the same encoder.
    """
    # Select categorical columns
    categorical_columns = train_df.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'class']

    print("Encoding the following categorical columns:")
    for col in categorical_columns:
        print(f"- {col}")

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train_data = encoder.fit_transform(train_df[categorical_columns])
    encoded_test_data = encoder.transform(test_df[categorical_columns])

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    # Convert encoded data to DataFrames
    encoded_train_df = pd.DataFrame(encoded_train_data, columns=encoded_columns, index=train_df.index)
    encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoded_columns, index=test_df.index)

    # Drop original categorical columns and concatenate encoded ones
    train_df = train_df.drop(columns=categorical_columns).join(encoded_train_df)
    test_df = test_df.drop(columns=categorical_columns).join(encoded_test_df)

    return train_df, test_df


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
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label='anomaly', zero_division=1)
        recall = recall_score(y_test, predictions, pos_label='anomaly', zero_division=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

def main(file_path):
    """
    Main function to load data, encode categorical features, and train models.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # One-hot encode categorical features
    train_df, test_df = encode_categorical_features(train_df, test_df)

    # Train and evaluate models
    train_and_evaluate_models(train_df, test_df)

# Run the program
if __name__ == "__main__":
    file_path = "Train_data.csv"  # Replace with the actual file path
    main(file_path)
