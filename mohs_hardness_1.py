import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score


'''
https://www.kaggle.com/competitions/playground-series-s3e25/overview
use regression to predict the Mohs hardness of a mineral, given its properties.
'''


hardness = pd.read_csv('data/mohs-hardness/train.csv')
hardness_test = pd.read_csv('data/mohs-hardness/test.csv')


# print(hardness.isna().sum())
# print(hardness_test.isna().sum())


correlation_matrix = hardness.corr()


# is 'allelectrons_Total' the atomic number ?
elec_min = hardness['allelectrons_Total'].min()
elec_max = hardness['allelectrons_Total'].max()


# print(correlation_matrix)

X_ = hardness.drop('Hardness', axis=1).to_numpy()
y_ = hardness['Hardness'].to_numpy().reshape([-1, 1])
X_ = X_.astype(float)
y_ = y_.astype(float)


X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=75/100)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float64), torch.tensor(y_train, dtype=torch.float64))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
in_dim = X_.shape[-1]


x_batch, y_batch = next(iter(train_dataloader))


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        # self.layer_1 = nn.Linear(in_dim, 32)
        # self.layer_2 = nn.Linear(32, 32)
        # self.layer_3 = nn.Linear(32, 64)
        # self.layer_4 = nn.Linear(64, 32)
        # self.layer_5 = nn.Linear(32, 1)
        # self.relu = nn.ReLU()
        self.sequence = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(p=3/4),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(p=3 / 4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.sequence(x)


regressor = Regressor()
optimizer = torch.optim.Adam(params=regressor.parameters(), lr=1/1_500)
criterion = nn.MSELoss()




