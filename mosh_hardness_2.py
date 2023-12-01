import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, median_absolute_error
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


# SCALING


feature_columns = filtered_train.drop(['id', 'Hardness', 'Dataset'], axis=1).columns.values
scaler = StandardScaler()
scaler.fit(filtered_train[feature_columns])
scaled_train_features = scaler.transform(filtered_train[feature_columns])
scaled_test_features = scaler.transform(test_data[feature_columns])
filtered_train[feature_columns] = scaled_train_features
test_data[feature_columns] = scaled_test_features


# MODEL EVALUATION


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor

linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
kr = KernelRidge()
dt = DecisionTreeRegressor()
svr = SVR()
knn = KNeighborsRegressor()
rf = RandomForestRegressor()
ada = AdaBoostRegressor()
lgbm_regressor = LGBMRegressor()
model_list = [linear, ridge, lasso, kr, dt, svr, knn, rf, ada, lgbm_regressor]
r2_results, mae_results = [], []

X_, y_ = filtered_train.drop(['id', 'Hardness', 'Dataset'], axis=1), filtered_train['Hardness']
X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=7/10, random_state=42)

for model in model_list:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    r2_results.append(r2)
    mae = median_absolute_error(y_test, predictions)
    mae_results.append(mae)
    print(f"model {str(model)} | r2 score: {r2:_.3f} | median abs error: {mae:_.3f}")


# OPTIMIZING HYPERPARAMS


param_grid = {
    'loss': ['squared_error', 'huber'],
    'learning_rate': [0.1, 0.05, 0.01, 0.001],
    'n_estimators': [100, 200, 500],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 3, 4],
    'max_depth': [3, 4, 8],
    'alpha': [0.8, 0.9, 0.99]
}
# model_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=4, verbose=1)
# model_search.fit(X_train, y_train)
# print('Best estimator:')
# print(model_search.best_estimator_)
# print('with score: %.6f' % model_search.best_score_)


