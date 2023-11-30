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
