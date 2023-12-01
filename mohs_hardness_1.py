import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
from utilities import Swish
from scipy.stats import zscore


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

hardness.drop('id', axis=1, inplace=True)
hardness.drop('allelectrons_Average', axis=1, inplace=True)  # features is really similar to 'atomicweight_Average


# deal with outliers
interested_columns = list(hardness.columns)[0:-1]
threshold = 5  # number of stds from 0 to be considered an outlier
z_scores = zscore(hardness[interested_columns])
filtered_train = hardness.loc[(z_scores < threshold).all(axis=1), hardness.columns]


# deal with skewness
filtered_skewed = filtered_train.iloc[:, 1:-2].skew()
# skewed_features = filtered_skewed[filtered_skew > 0.75].index.values
skewed_features = filtered_skewed[filtered_skewed.abs() > 0.75].index.values
filtered_train[skewed_features] = np.log1p(filtered_train[skewed_features])  # np.log1p(x) = np.log(1+x)
hardness_test[skewed_features] = np.log1p(hardness_test[skewed_features])


# X_ = hardness.drop('Hardness', axis=1).to_numpy()
# y_ = hardness['Hardness'].to_numpy().reshape([-1, 1])
X_ = filtered_train.drop('Hardness', axis=1).to_numpy()
y_ = filtered_train['Hardness'].to_numpy().reshape([-1, 1])
X_ = X_.astype(float)
y_ = y_.astype(float)


X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=80/100)
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
            nn.Linear(in_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(p=1/10),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Dropout(p=1/10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.sequence(x)


class SwishRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.layer_1 = nn.Linear(in_dim, 32)
        self.norm_1 = nn.BatchNorm1d(32)
        self.layer_2 = nn.Linear(32, 32)
        self.norm_2 = nn.BatchNorm1d(32)
        self.layer_3 = nn.Linear(32, 64)
        self.norm_3 = nn.BatchNorm1d(64)
        self.layer_4 = nn.Linear(64, 32)
        self.norm_4 = nn.BatchNorm1d(32)
        self.layer_5 = nn.Linear(32, 1)
        self.swish = Swish()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=1/10)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        # print(x)
        x = self.norm_1(x)
        x = self.relu(self.layer_2(x))
        x = self.norm_2(x)
        x = self.relu(self.layer_3(x))
        x = self.norm_3(x)
        x = self.dropout(x)
        x = self.relu(self.layer_4(x))
        x = self.norm_4(x)
        x = self.dropout(x)
        x = self.layer_5(x)
        return x


regressor = SwishRegressor()
optimizer = torch.optim.Adam(params=regressor.parameters(), lr=1/1_500)
criterion = nn.MSELoss()


for epoch in range(1, 1_001):
    total_loss = 0
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        regressor.train()
        preds = regressor(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().numpy()
    if epoch % 10 == 0:
        # print("Epoch %d | Loss %.4f" % (epoch, total_loss))
        print(f'Epoch {epoch} | Loss {total_loss:_.3f}')


torch.save(regressor, '/home/fabio/PycharmProjects/kaggle-challenges/data/models/mohs_regressor.pth')


# regressor = torch.load('/home/fabio/PycharmProjects/kaggle-challenges/data/models/mohs_regressor.pth')


predictions = regressor(torch.tensor(X_test, dtype=torch.float64)).detach()
y_test = torch.tensor(y_test, dtype=torch.float64)
print(f'R2 score : {r2_score(y_test, predictions)}')

# X_validation = torch.tensor(hardness_test.to_numpy().astype(float), dtype=torch.float64)
# predictions = regressor(X_validation).detach().squeeze()
# hardness_v = pd.read_csv('data/mohs-hardness/test.csv')
# submission = pd.DataFrame({'Id': hardness_v['Id'], 'Hardness': predictions})
# submission.to_csv('data/mohs-hardness/submission-1st.csv', index=False)


# 0.37

