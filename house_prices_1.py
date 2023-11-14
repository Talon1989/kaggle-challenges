import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd


'''
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
Predict the sales price for each house.
For each Id in the test set, you must predict the value of the SalePrice variable
'''


house_prices = pd.read_csv('data/house-prices/train.csv')
house_prices_validation = pd.read_csv('data/house-prices/test.csv')




