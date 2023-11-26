import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score


# IMPLEMENTATION WITH NN


house_prices = pd.read_csv('data/house-prices/train.csv')
house_prices_validation = pd.read_csv('data/house-prices/test.csv')
house_prices_total = [house_prices, house_prices_validation]


garage_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
features_to_drop = [feat for feat in garage_features if feat != 'GarageArea']
for df in house_prices_total:
    df.drop('LotFrontage', axis=1, inplace=True)
    df.drop(features_to_drop, axis=1, inplace=True)
    df.drop('Id', axis=1, inplace=True)
    df.drop('FireplaceQu', axis=1, inplace=True)  # already have similar data
    df.drop('PoolQC', axis=1, inplace=True)  # too specific for too few elements
    df.drop('MoSold', axis=1, inplace=True)  # useless feature
    df.drop('Utilities', axis=1, inplace=True)  # all utilities in validation
    df.drop('Condition1', axis=1, inplace=True)  # non significant
    df.drop('Condition2', axis=1, inplace=True)  # non significant
    df.drop('BldgType', axis=1, inplace=True)  # non significant
    df.drop('HouseStyle', axis=1, inplace=True)  # CONSIDER implementing it
    df.drop('OverallCond', axis=1, inplace=True)  # OverallQual is too similar
    df.drop('YearBuilt', axis=1, inplace=True)  # Keep only YearRemodAdd
    df.drop('RoofMatl', axis=1, inplace=True)  # few data in validation
    df.drop('Exterior1st', axis=1, inplace=True)
    df.drop('Exterior2nd', axis=1, inplace=True)
    df.drop('MasVnrType', axis=1, inplace=True)  # too many missing data
    df.drop('MasVnrArea', axis=1, inplace=True)  # too many missing data
    df.drop('ExterCond', axis=1, inplace=True)  # non significant
    df.drop('BsmtQual', axis=1, inplace=True)  # non significant
    df.drop('BsmtExposure', axis=1, inplace=True)  # non significant
    df.drop('BsmtFinType1', axis=1, inplace=True)  # non significant
    df.drop('BsmtFinSF1', axis=1, inplace=True)  # non significant
    df.drop('BsmtFinType2', axis=1, inplace=True)  # non significant
    df.drop('BsmtFinSF2', axis=1, inplace=True)  # non significant
    df.drop('BsmtUnfSF', axis=1, inplace=True)  # non significant
    df.drop('TotalBsmtSF', axis=1, inplace=True)  # non significant
    df.drop('Heating', axis=1, inplace=True)  # non significant
    df.drop('LowQualFinSF', axis=1, inplace=True)  # non significant
    df.drop('PavedDrive', axis=1, inplace=True)  # non significant
    df.drop('MiscVal', axis=1, inplace=True)  # non significant
    df.drop('YrSold', axis=1, inplace=True)  # non significant
    df.drop('SaleType', axis=1, inplace=True)  # non significant
    df.drop('MSSubClass', axis=1, inplace=True)  # non significant


for df in house_prices_total:
    df['Alley'].fillna('None', inplace=True)
    df['MiscFeature'].fillna('None', inplace=True)
    df.rename(columns={
        'BedroomAbvGr': 'Bedroom',
        'KitchenAbvGr': 'Kitchen'}, inplace=True)


# validation data has no Industrial, Commercial or Agricultural zoning classification (MSZoning)
# join all residential together (non-floating village)
house_prices_validation['MSZoning'].fillna('RL', inplace=True)
for df in house_prices_total:
    df.loc[df['MSZoning'].str.startswith('R'), 'MSZoning'] = 'Res'
    df.loc[df['MSZoning'].str.startswith('F'), 'MSZoning'] = 'Fl'
    df.loc[df['MSZoning'].str.startswith('C'), 'MSZoning'] = 'Comm'


# LotShape: join all irregular 2 and 3 together
for df in house_prices_total:
    df.loc[df['LotShape'] == 'IR3', 'LotShape'] = 'IR2'


above_avg = ['Edwards', 'CollgCr', 'Blueste', 'NAmes', 'NoRidge',
        'NridgHt', 'NWAmes', 'Somerst', 'StoneBr', 'Timber', 'Veenker']
less_avg = ['Blmngtn', 'BrDale', 'OldTown', 'Crawfor', 'Mitchel', 'BrkSide',
          'Gilbert', 'MeadowV', 'NPkVill', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW']
river = ['ClearCr']
iowa_road = ['IDOTRR']

indices = house_prices.loc[house_prices['Neighborhood'].isin(above_avg)].index

for df in house_prices_total:
    df.loc[df['Neighborhood'].isin(above_avg), 'Neighborhood'] = 'Rich'
    df.loc[df['Neighborhood'].isin(less_avg), 'Neighborhood'] = 'Medium'
    df.loc[df['Neighborhood'].isin(river), 'Neighborhood'] = 'River'
    df.loc[df['Neighborhood'].isin(iowa_road), 'Neighborhood'] = 'Road'


# dealing with basement, remove all conditions and quality, just boolean True or False
for df in house_prices_total:
    non_nan_indices = df[df['BsmtCond'].notna()].index
    df.loc[non_nan_indices, 'BsmtCond'] = 'Y'
    df['BsmtCond'].fillna('N', inplace=True)
    df.rename(columns={'BsmtCond': 'Basement'}, inplace=True)


# only nan training dataset
nan_index = house_prices[house_prices['Electrical'].isna()].index
house_prices.drop(axis=0, index=nan_index, inplace=True)
# join fuse types of Electrical together
for df in house_prices_total:
    df.loc[df['Electrical'].str.startswith('F'), 'Electrical'] = 'Fuse'


# dealing with nan
for df in house_prices_total:
    df['BsmtFullBath'].fillna(0, inplace=True)
    df['BsmtHalfBath'].fillna(0, inplace=True)
    df['KitchenQual'].fillna('TA', inplace=True)
    df['Functional'].fillna('Typ', inplace=True)


# joining functional together
for df in house_prices_total:
    df.loc[df['Functional'].str.startswith('Mi'), 'Functional'] = 'Min'
    df.loc[df['Functional'].str.startswith('Mo'), 'Functional'] = 'Mod'
    df.loc[df['Functional'].str.startswith('Ma'), 'Functional'] = 'Mod'
    df.loc[df['Functional'].str.startswith('S'), 'Functional'] = 'Sev'


# SIMPLIFYING FENCE
for df in house_prices_total:
    no_fence_indices = df[df['Fence'].isna()].index
    yes_fence_indices = df[df['Fence'].isna() == False].index
    df.loc[no_fence_indices, 'Fence'] = 0
    df.loc[yes_fence_indices, 'Fence'] = 1


# only tennis court in training dataset and no tennis court in validation
tennis_court_index = house_prices[house_prices['MiscFeature'] == 'TenC'].index
house_prices.drop(tennis_court_index, inplace=True)


# validation has 1 nan on 'GarageArea'
house_prices_validation['GarageArea'].fillna(value=0., inplace=True)


categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LotConfig', 'LandContour',
                       'LandSlope', 'Neighborhood', 'RoofStyle', 'ExterQual',
                       'Foundation', 'Basement', 'HeatingQC', 'CentralAir', 'Electrical',
                       'KitchenQual', 'Functional', 'MiscFeature', 'SaleCondition']
combined_datasets = pd.concat([house_prices, house_prices_validation], axis=0)
combined_datasets = pd.get_dummies(combined_datasets, columns=categorical_columns)
scaler = StandardScaler()
scaler.fit(combined_datasets.drop('SalePrice', axis=1).to_numpy().astype(float))  # NOT IMPLEMENTED
house_prices = combined_datasets.iloc[0:house_prices.shape[0], :].copy()
house_prices_validation = combined_datasets.iloc[house_prices.shape[0]:, :].copy()
house_prices_validation.drop('SalePrice', axis=1, inplace=True)


X_ = house_prices.drop('SalePrice', axis=1).to_numpy()
y_ = house_prices['SalePrice'].to_numpy().reshape([-1, 1])
X_ = X_.astype(float)
y_ = y_.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=75/100)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float64), torch.tensor(y_train, dtype=torch.float64))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
in_dim = X_.shape[-1]


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
        self.mod = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mod(x)


regressor = Regressor()
optimizer = torch.optim.Adam(params=regressor.parameters(), lr=1/1_500)
criterion = nn.MSELoss()
# x_b, y_b = next(iter(train_dataloader))


for epoch in range(1, 6_001):
    total_loss = 0
    for x_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        regressor.train()
        preds = regressor(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().numpy()
    if epoch % 20 == 0:
        # print("Epoch %d | Loss %.4f" % (epoch, total_loss))
        print(f'Epoch {epoch} | Loss {total_loss:_}')





# house_prices_v = pd.read_csv('data/house-prices/test.csv')





