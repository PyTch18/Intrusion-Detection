import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from distfit import distfit
from matplotlib import pyplot as plt
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

def z_score(df1, thresholds1):

    for column in df1.columns:
        if df1.dtypes[column] in ['int64', 'float64']: # getting the z-score for the numerical data
            mean = df1[column].mean() # Mean for the data
            std = df1[column].std() # Standard deviation for the data
            z_scores = (df1[column] - mean) / std # Z-score for the data

            for threshold in thresholds1:
                anomalies = z_scores[abs(z_scores) > threshold]
                print(f"{column} with higher threshold than {threshold}:\n {anomalies}")

thresholds = [1.5,2.0,2.5,3.0]
z_score(df, thresholds)

