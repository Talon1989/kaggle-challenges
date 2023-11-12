import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


titanic_data = pd.read_csv('data/titanic/train.csv')
validation_data = pd.read_csv('data/titanic/test.csv')


def analysis_of_missing_data(data: pd.DataFrame):
    total = data.isna().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
    ms = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
    ms = ms[ms['Percentage'] > 0]
    print('Missing data:')
    print(ms)
    # categories = ['Cabin', 'Age', 'Embarked']
    plt.bar(ms.index, ms['Percentage'], color='C0')
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Missing Values %', fontsize=15)
    plt.title('Missing data by feature')
    plt.show()
    plt.clf()


# analysis_of_missing_data(titanic_data)


print(validation_data['Age'].mean())
print(validation_data['Embarked'].mode()[0])  # .mode returns most common value


# FILLING MISSING VALUES ON VALIDATION DATASET
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
validation_data['Age'].fillna(validation_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
validation_data['Embarked'].fillna(validation_data['Embarked'].mode()[0], inplace=True)
validation_data['Fare'].fillna(validation_data['Fare'].median(), inplace=True)


# SINCE CABIN FEATURE IS MISSING MOST OF DATA IN BOTH TRAIN AND VALIDATION WE DROP IT
titanic_data.drop(['Cabin'], axis=1, inplace=True)
validation_data.drop(['Cabin'], axis=1, inplace=True)




























































































































