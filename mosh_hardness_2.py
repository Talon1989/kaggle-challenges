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


'''
https://www.kaggle.com/competitions/playground-series-s3e25/overview
use regression to predict the Mohs hardness of a mineral, given its properties.
try using a classifier from 0 to 10 with 0.5 increments
'''


hardness = pd.read_csv('data/mohs-hardness/train.csv')
hardness_test = pd.read_csv('data/mohs-hardness/test.csv')


# print(hardness.isna().sum())
# print(hardness_test.isna().sum())


correlation_matrix = hardness.corr()


# encoder = LabelEncoder()
# encoder.fit(np.arange(0, 10, 0.5))
def round_to_nearest_half(value):
    return round(value * 2) / 2


hardness['Hardness'].apply(round_to_nearest_half)


X_ = hardness.drop('Hardness', axis=1).to_numpy()
y_ = hardness['Hardness'].to_numpy().reshape([-1, 1])
X_ = X_.astype(float)
y_ = y_.astype(float)


X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=75/100)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float64), torch.tensor(y_train, dtype=torch.float64))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
in_dim = X_.shape[-1]
