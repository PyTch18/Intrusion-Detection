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
            dist.fit_transform(df1[column].dropna())
            dist.plot()
            print(f"Best distribution for '{column}': {dist.model}")

best_fit_distribution(df)