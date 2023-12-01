import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import zscore
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


'''
https://www.kaggle.com/competitions/playground-series-s3e25/overview
use regression to predict the Mohs hardness of a mineral, given its properties.
'''


train_data = pd.read_csv('data/mohs-hardness/train.csv')
test_data = pd.read_csv('data/mohs-hardness/test.csv')


# print(hardness.isna().sum())
# print(hardness_test.isna().sum())


correlation_matrix = train_data.corr()
skewness = train_data.skew()


train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
variables = [feature for feature in train_data.columns if feature not in ['id', 'Hardness', 'Dataset']]


def create_variable_plots(variable):
    custom_palette = ['#3498db', '#e74c3c']
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, test_data], axis=0), x=variable, y="Dataset")
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN & TEST]")
    plt.legend()
    # Adjust spacing between subplots
    plt.tight_layout()
    # Show the plots
    plt.show()


# for v in variables:
#     create_variable_plots(v)


# create_variable_plots('Hardness')


# 5 highest hardness correlated features
important_features = correlation_matrix['Hardness'].abs().sort_values(ascending=False).index[1:6]
important_features = list(important_features)


# DEALING WITH OUTLIERS IN TRAINING DATA
# z-score returns the number of stds from the mean of a standard gaussian: Z = (x-mean)/std

interested_columns = list(train_data.columns)[1:-2]
threshold = 5  # number of stds from 0 to be considered an outlier
z_scores = zscore(train_data[interested_columns])
filtered_train = train_data.loc[(z_scores < threshold).all(axis=1), train_data.columns]
# z_scores_elec_total = zscore(train_data[interested_columns[0]])
# filtered_data = train_data[(z_scores_elec_total < threshold)]


# TRANSFORMATION OF SKEWED DATA
# need to remove id and Dataset to get numerical skew, also remove hardness

filtered_skewed = filtered_train.iloc[:, 1:-2].skew()
# skewed_features = filtered_skewed[filtered_skew > 0.75].index.values
skewed_features = filtered_skewed[filtered_skewed.abs() > 0.75].index.values
filtered_train[skewed_features] = np.log1p(filtered_train[skewed_features])  # np.log1p(x) = np.log(1+x)

# do it for the test (same features are skewed)
test_data[skewed_features] = np.log1p(test_data[skewed_features])















