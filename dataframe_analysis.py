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
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# lin = np.arange(10) * 2
# pol = np.arange(10) ** 2
# exp = 2 ** np.arange(10)
# fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
# axes[0].plot(lin)
# axes[0].set_title('Linear')
# axes[1].plot(pol)
# axes[1].set_title('Polynomial')
# axes[2].plot(exp)
# axes[2].set_title('Exponential')
# plt.show()
# plt.clf()


dataframe = pd.read_csv('data/house-prices/train.csv')
df_2 = pd.read_csv('data/house-prices/test.csv')


class DataVisualization:
    def __init__(self, df: pd.DataFrame, target_feature: str=None):
        self.df = df
        self.df_size = len(self.df) * len(self.df.columns)
        self.nan_elements = self.df.isna()
        self.target_feature = target_feature
        target_index = np.where(self.df.columns.values == self.target_feature)[0]
        self.feature_columns = np.delete(self.df.columns.values, target_index)
        self.categorical_features = self.df[self.feature_columns].select_dtypes(include=['object']).columns.values
        self.numerical_features = self.df[self.feature_columns].select_dtypes(include=['number']).columns.values

    def feature_types(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns.values
        numerical_columns = self.df.select_dtypes(include=['number']).columns.values
        print('-----------------------------------------------------------')
        print(f"n categorical columns: {len(categorical_columns)}")
        print(f"n numerical columns: {len(numerical_columns)}")
        print(f'total number of nan values in the {self.df_size:_} '
              f'element dataframe: {self.nan_elements.sum().sum():_}')
        nan_columns = self.nan_elements.columns.values
        print(f"n columns with nan values: {len(nan_columns)}")
        nan_categorical_columns = list(set(nan_columns) & set(categorical_columns))
        nan_numerical_columns = list(set(nan_columns) & set(numerical_columns))
        print('-----------------------------------------------------------')
        print(f"n categorical columns with nan elements: {len(nan_categorical_columns)}")
        print(f"n numerical columns with nan elements: {len(nan_numerical_columns)}")

    def correlation_matrix(self):
        corr = self.df[self.numerical_features].corr()
        plt.figure(figsize=(8, 6))
        annot_kws = {"size": 8, "rotation": 45}
        # annot_kws = {"size": 10}
        mask = np.triu(np.ones_like(corr, dtype=bool))
        heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=1/2,
                              fmt='.2f', annot_kws=annot_kws, mask=mask)
        plt.title('Correlation Matrix')
        plt.show()
        plt.clf()

    def skewness(self):
        skew = self.df[self.numerical_features].skew().sort_values(ascending=False)
        sns.barplot(x=skew, y=skew.index, palette='viridis')
        plt.title('Skewness')
        plt.xlabel('Value')
        plt.ylabel('Skewness')
        plt.show()

    def feature_plots(self, feature):
        assert feature in self.feature_columns
        custom_palette = ['#3498db', '#e74c3c']
        sns.set_style('whitegrid')

        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x=feature)
        plt.xlabel(feature)
        plt.title(f"Box Plot for {feature}")

        plt.subplot(1, 2, 2)
        sns.histplot(data=self.df, x=feature, color=custom_palette[0], kde=True, bins=30)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Histogram for {feature}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dataframe_feature_histogram(dataframe_1, dataframe_2, feature):
        custom_palette = ['#3498db', '#e74c3c']
        sns.set_style('whitegrid')
        sns.histplot(data=dataframe_1, x=feature,
                     color=custom_palette[0], kde=True, bins=30, label='df_1')
        sns.histplot(data=dataframe_2, x=feature,
                     color=custom_palette[1], kde=True, bins=30, label='df_2')
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Histogram for {feature}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dataframe_features_histogram(dataframe_1, dataframe_2, features):
        custom_palette = ['#3498db', '#e74c3c']
        sns.set_style('whitegrid')
        num_features = len(features)
        num_cols = int(np.ceil(np.sqrt(num_features)))
        num_rows = int(np.ceil(num_features / num_cols))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        # fig, axs = plt.subplots(nrows=len(features), ncols=1)
        axs = axs.ravel()  # Flatten the array of axes
        for i, feature in enumerate(features):
            sns.histplot(data=dataframe_1, x=feature, color=custom_palette[0], kde=True, bins=30, label='df_1',
                         ax=axs[i])
            sns.histplot(data=dataframe_2, x=feature, color=custom_palette[1], kde=True, bins=30, label='df_2',
                         ax=axs[i])
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel("Frequency")
            axs[i].set_title(f"Histogram for {feature}")
            axs[i].legend()
        plt.tight_layout()
        plt.show()


visualization = DataVisualization(dataframe, 'SalePrice')
# visualization.correlation_matrix()
# visualization.feature_plots('LotArea')
# visualization.dataframe_feature_histogram(dataframe, df_2, 'LotArea')
features = ['LotArea', 'Street', 'Alley', 'OverallCond', 'OverallQual', 'BedroomAbvGr', 'OpenPorchSF']
# visualization.dataframe_features_histogram(dataframe, df_2, features)
visualization.skewness()




