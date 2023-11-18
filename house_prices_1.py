import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd


'''
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
Predict the sales price for each house.
For each Id in the test set, you must predict the value of the SalePrice variable
'''


house_prices = pd.read_csv('data/house-prices/train.csv')
house_prices_validation = pd.read_csv('data/house-prices/test.csv')
house_prices_total = [house_prices, house_prices_validation]


# MAIN IDEAS  ----------------------------------------------------------------------------------


'''
one-hot encoding:  pd.get_dummies(DataFrame, column_name)
label encoder:  DataFrame[col_name] = LabelEncoder().fit_transform(DataFrame[col_name]) 
'''


# DATA PREPROCESSING  -----------------------------------------------------------------------------------


alleys = ['None', 'Grvl', 'Pave']


for df in house_prices_total:
    df['Alley'].fillna('None', inplace=True)
    df['MiscFeature'].fillna('None', inplace=True)

# alley_encoder = LabelEncoder()
# alley_encoder.fit(alleys)
# house_prices['Alley'] = alley_encoder.transform(house_prices['Alley'])


#  DEALING WITH MISSING LOT FRONTAGE DATA (IMPUTATION ON LOT AREA)
#  DROPPING IMPUTATION SINCE LOSS IS TOO HIGH

for df in house_prices_total:
    # lot_frontage_nan_indices = np.where(df['LotFrontage'].isna())[0]
    # lot_frontages_areas = df[['LotFrontage', 'LotArea']].copy()
    # lot_frontages_areas.dropna(axis=0, inplace=True)
    # regressor = LinearRegression()
    # regressor.fit(  # only trained on train.csv
    #     lot_frontages_areas['LotArea'].to_numpy().reshape([-1, 1]),
    #     lot_frontages_areas['LotFrontage'].to_numpy().reshape([-1, 1])
    # )
    # # loss = regressor.score(
    # #     lot_frontages_areas['LotArea'].to_numpy().reshape([-1, 1]),
    # #     lot_frontages_areas['LotFrontage'].to_numpy().reshape([-1, 1]))
    # # print(loss)
    # predictions = regressor.predict(
    #     df.loc[lot_frontage_nan_indices]['LotArea'].to_numpy().reshape([-1, 1])
    # )
    # df.loc[lot_frontage_nan_indices, 'LotFrontage'] = predictions.squeeze()
    df.drop('LotFrontage', axis=1, inplace=True)


# overall view of missing data (and none correspondence)
# whatever we need and is missing we use the average of all the others


train_missing_data = house_prices.isna().sum(axis=0)
validation_missing_data = house_prices_validation.isna().sum(axis=0)


mean_lot_area = house_prices['LotArea'].mean()
house_prices['LotArea'].fillna(mean_lot_area)
house_prices.drop(np.where(house_prices['Electrical'].isna())[0], axis=0, inplace=True)  # eliminating nan row


def categorical_data_analysis():
    categorical_columns = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'ExterQual', 'ExterCond',
        'Foundation', 'Heating', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'SaleType', 'SaleCondition'
    ]
    index_of_nan = np.where(house_prices['Electrical'].isna())[0]
    house_prices.drop(index_of_nan, axis=0, inplace=True)
    for c in categorical_columns:
        print(c)
        print(np.unique(house_prices[c]))
        print()


# DEALING WITH GARAGE: keep only GarageCars representative of 'garage' feature
# AND FIREPLACE

garage_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
features_to_drop = [feat for feat in garage_features if feat != 'GarageCars']
for df in house_prices_total:
    df.drop(features_to_drop, axis=1, inplace=True)
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
    df.drop('MasVnrType', axis=1, inplace=True)  # non significant
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


# DEALING WITH PORCH

# using training data mean values also for validation
open_porch_mean = house_prices['OpenPorchSF'].mean()
enclosed_porch_mean = house_prices['EnclosedPorch'].mean()
porch_means = [open_porch_mean, enclosed_porch_mean]
porch_types = ['OpenPorchSF', 'EnclosedPorch']

for i in range(len(porch_types)):
    for df in house_prices_total:
        no_indices = df[df[porch_types[i]] == 0].index
        small_indices = df[(0 < df[porch_types[i]]) & (df[porch_types[i]] < porch_means[i])].index
        big_indices = df[df[porch_types[i]] >= open_porch_mean].index
        df[porch_types[i]] = pd.Categorical(df[porch_types[i]], categories=['Zero', 'Small', 'Big'])
        df.loc[no_indices, porch_types[i]] = 'Zero'
        df.loc[small_indices, porch_types[i]] = 'Small'
        df.loc[big_indices, porch_types[i]] = 'Big'

for porch in ['3SsnPorch', 'ScreenPorch']:
    for df in house_prices_total:
        non_zero_indices = df[df[porch] != 0].index
        df.loc[non_zero_indices, porch] = 1

# SIMPLIFYING POOL
for df in house_prices_total:
    pool_non_zero_indices = df[df['PoolArea'] != 0].index
    # df.drop(columns=['PoolArea'])  # dropping 'PoolArea' and creating boolean 'Pool' with values based on indices
    # df['Pool'] = 0
    # df.loc[non_zero_indices, 'Pool'] = 1
    df.loc[pool_non_zero_indices, 'PoolArea'] = 1  # replacing values and renaming
    df.rename(columns={'PoolArea': 'Pool'}, inplace=True)

# SIMPLIFYING FENCE
for df in house_prices_total:
    no_fence_indices = df[df['Fence'].isna()].index
    yes_fence_indices = df[df['Fence'].isna() == False].index
    df.loc[no_fence_indices, 'Fence'] = 0
    df.loc[yes_fence_indices, 'Fence'] = 1


# no_open_porch_indices = house_prices[house_prices['OpenPorchSF'] == 0].index
# small_open_porch_indices = house_prices[(0 < house_prices['OpenPorchSF']) & (house_prices['OpenPorchSF'] < open_porch_mean)].index
# big_open_porch_indices = house_prices[house_prices['OpenPorchSF'] >= open_porch_mean].index
# house_prices.loc[no_open_porch_indices, 'OpenPorchSF'] = 'Zero'
# house_prices.loc[small_open_porch_indices, 'OpenPorchSF'] = 'Small'
# house_prices.loc[big_open_porch_indices, 'OpenPorchSF'] = 'Big'


# check two lists contain no common elements
# print(set(small_open_porch_indices).isdisjoint(set(big_open_porch_indices)))


# CATEGORICAL PREPROCESSING
# h = pd.get_dummies(house_prices, columns=['MSZoning, LotShape', 'LotConfig', 'LandSlope'])
categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LotConfig', 'LandSlope', 'LandContour']

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


# Consider LandContour: join non-levelled properties together
# for df in house_prices_total:
#     df.loc[df['LandContour'] != 'Lvl', 'LandContour'] = 'Non-Lvl'


# OverallQual and OverallCond are extemely positively correlated, so we just use one and simplify the data
# corr = house_prices[['OverallQual', 'OverallCond']].corr()
for df in house_prices_total:
    low_indices = df[df['OverallQual'] <= 4].index
    medium_indices = df[(5 <= df['OverallQual']) & (df['OverallQual'] <= 7)].index
    high_indices = df[df['OverallQual'] >= 8].index
    df['OverallQual'] = pd.Categorical(df['OverallQual'], categories=['Low', 'Medium', 'High'])
    df.loc[low_indices, 'OverallQual'] = 'Low'
    df.loc[medium_indices, 'OverallQual'] = 'Medium'
    df.loc[high_indices, 'OverallQual'] = 'High'

for df in house_prices_total:
    pre_war_indices = df[df['YearRemodAdd'] <= 1945].index
    old_indices = df[(df['YearRemodAdd'] >= 1946) & (df['YearRemodAdd'] <= 1975)].index
    recent_indices = df[(df['YearRemodAdd'] >= 1976) & (df['YearRemodAdd'] <= 1999)].index
    modern_indices = df[df['YearRemodAdd'] >= 2000].index
    df.rename(columns={'YearRemodAdd': 'Year'}, inplace=True)
    df['Year'] = pd.Categorical(df['Year'], categories=['Pre-War', 'Old', 'Recent', 'Modern'])
    df.loc[pre_war_indices, 'Year'] = 'Pre-War'
    df.loc[old_indices, 'Year'] = 'Old'
    df.loc[recent_indices, 'Year'] = 'Recent'
    df.loc[modern_indices, 'Year'] = 'Modern'


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


# join fuse types of Electrical together
for df in house_prices_total:
    df.loc[df['Electrical'].str.startswith('F'), 'Electrical'] = 'Fuse'


# TODO: 1stFlrSF ->











































