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


# DATA PREPROCESSING  -----------------------------------------------------------------------------------


alleys = ['None', 'Grvl', 'Pave']


for df in house_prices_total:
    # df.drop('Alley', axis=1, inplace=True)  # too many nan
    # df.drop('LotFrontage', axis=1, inplace=True)  # consider not dropping this
    df['Alley'].fillna('None', inplace=True)

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







