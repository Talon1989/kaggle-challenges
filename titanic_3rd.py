import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict


'''
https://www.kaggle.com/code/vinothan/titanic-model-with-90-accuracy
'''


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


# print(validation_data['Age'].mean())
# print(validation_data['Embarked'].mode()[0])  # .mode returns most common value


# FILLING MISSING VALUES ON VALIDATION DATASET
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
validation_data['Age'].fillna(validation_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
validation_data['Embarked'].fillna(validation_data['Embarked'].mode()[0], inplace=True)
validation_data['Fare'].fillna(validation_data['Fare'].median(), inplace=True)


# SINCE CABIN FEATURE IS MISSING MOST OF DATA IN BOTH TRAIN AND VALIDATION WE DROP IT
titanic_data.drop(['Cabin'], axis=1, inplace=True)
validation_data.drop(['Cabin'], axis=1, inplace=True)


def title_formatter(data: pd.DataFrame):
    import re
    def get_title(name):
        title = re.search(' ([A-Za-z]+)\.', name)
        if title:
            return title.group(1)
        return ""
    data['Title'] = data['Name'].apply(get_title)
    data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        value='Rare',
        inplace=True)
    # data['Title'].replace('Mlle', 'Miss', inplace=True)
    # data['Title'].replace('Ms', 'Miss', inplace=True)
    # data['Title'].replace('Mme', 'Mrs', inplace=True)
    data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}, inplace=True)


all_data = [titanic_data, validation_data]


for df in all_data:  # combine SibSp and Parch
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    title_formatter(df)
    df['Age_bin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 20, 40, 120],
        labels=['Children', 'Teenager', 'Adult', 'Elder'])
    df['Fare_bin'] = pd.cut(
        df['Fare'],
        bins=[0, 7.91, 14.45, 31, 120],
        labels=['Low_fare', 'Median_fare', 'Average_fare', 'High_fare'])
    # df.drop(['Name', 'SibSp', 'Parch', 'Age', 'Fare', 'Ticket'], axis=1, inplace=True)
    df.drop(['Name', 'Age', 'Fare', 'Ticket'], axis=1, inplace=True)


titanic_data.drop('PassengerId', axis=1, inplace=True)  # need id for validation data
#  tricky to use .get_dummies inside the loop
titanic_data = pd.get_dummies(
    titanic_data,
    columns=["Sex", "Title", "Age_bin", "Embarked", "Fare_bin"],
    prefix=["Sex", "Title", "Age_type", "Em_type", "Fare_type"])
validation_data = pd.get_dummies(
    validation_data,
    columns=["Sex", "Title", "Age_bin", "Embarked", "Fare_bin"],
    prefix=["Sex", "Title", "Age_type", "Em_type", "Fare_type"])


def print_corr_matrix():
    sns.heatmap(titanic_data.corr(), annot=True, cmap='RdYlGn', linewidths=.2)
    # fig = plt.gcf()
    # fig.set_size_inches(20, 12)
    # plt.show()


# ENSEMBLE OF METHODS


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


titanic_feature_data = titanic_data.drop('Survived', axis=1)
titanic_target_data = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(titanic_feature_data, titanic_target_data,
                                                    train_size=7/10, random_state=42)


model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
prediction_lr = model_lr.predict(X_test)
print('Accuracy of Logistic Regression: %.4f' % accuracy_score(prediction_lr, y_test))


model_rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, min_samples_leaf=1,
                                  max_features=2, oob_score=True, random_state=1, n_jobs=1)
model_rf.fit(X_train, y_train)
prediction_rf = model_rf.predict(X_test)
print('Accuracy of Random Forest: %.4f' % accuracy_score(prediction_rf, y_test))


model_svm = SVC()
model_svm.fit(X_train, y_train)
prediction_svm = model_svm.predict(X_test)
print('Accuracy of SVM: %.4f' % accuracy_score(prediction_svm, y_test))


model_knn = KNeighborsClassifier(n_neighbors=4)
model_knn.fit(X_train, y_train)
prediction_knn = model_knn.predict(X_test)
print('Accuracy of KNN: %.4f' % accuracy_score(prediction_knn, y_test))


model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
prediction_gnb = model_gnb.predict(X_test)
print('Accuracy of GNB: %.4f' % accuracy_score(prediction_gnb, y_test))


model_dt = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=1, max_features=2)
model_dt.fit(X_train, y_train)
prediction_dt = model_dt.predict(X_test)
print('Accuracy of Decision Tree: %.4f' % accuracy_score(prediction_dt, y_test))


model_ada = AdaBoostClassifier()
model_ada.fit(X_train, y_train)
prediction_ada = model_ada.predict(X_test)
print('Accuracy of Ada Boos: %.4f' % accuracy_score(prediction_ada, y_test))


model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train, y_train)
prediction_lda = model_lda.predict(X_test)
print('Accuracy of Linear Discriminat Analysis: %.4f' % accuracy_score(prediction_lda, y_test))


model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, y_train)
prediction_gb = model_gb.predict(X_test)
print('Accuracy of Gradient Boosting: %.4f' % accuracy_score(prediction_gb, y_test))


# cross validations
cross_validation_lr = cross_val_score(model_lr, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_rf = cross_val_score(model_rf, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_svm = cross_val_score(model_svm, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_knn = cross_val_score(model_knn, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_gnb = cross_val_score(model_gnb, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_dt = cross_val_score(model_dt, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_ada = cross_val_score(model_ada, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_lda = cross_val_score(model_lda, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')
cross_validation_gb = cross_val_score(model_gb, titanic_feature_data, titanic_target_data,
                                      cv=10, scoring='accuracy')


model_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Support Vector Machines',
              'K-Nearest Neighbours', 'Gaussian NB', 'Decision Tree', 'ADA Boost',
              'Linear Discriminant Analysis', 'Gradient Boost'],
    'Score': [
        cross_validation_lr.mean(), cross_validation_rf.mean(), cross_validation_svm.mean(),
        cross_validation_knn.mean(), cross_validation_gnb.mean(), cross_validation_dt.mean(),
        cross_validation_ada.mean(), cross_validation_lda.mean(), cross_validation_gb.mean()
    ]
})
model_results.sort_values(by='Score', ascending=False, inplace=True)


# TUNING BEST MODEL (RANDOM FOREST)


import xgboost as xgb


X_train = titanic_feature_data.copy()
y_train = titanic_target_data.copy()
X_test = validation_data.drop('PassengerId', axis=1).copy()


# k_fold = KFold(n_splits=10)  # splits data into 10 equal parts
# model = GradientBoostingClassifier()
# param_grid = {
#     'loss': ["log_loss"],
#     'n_estimators': [100, 200, 300, 400],
#     'learning_rate': [0.1, 0.05, 0.01, 0.001],
#     'max_depth': [4, 8],
#     'min_samples_leaf': [100, 150],
#     'max_features': [0.3, 0.2, 0.1]
# }
# model_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=4, verbose=1)
# model_search.fit(X_train, y_train)
# print('Best estimator:')
# print(model_search.best_estimator_)
# print('with score: %.6f' % model_search.best_score_)


random_forest_model = RandomForestClassifier(
    bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features=2,
    max_leaf_nodes=None, min_impurity_decrease=0., min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0., n_estimators=400, n_jobs=1, oob_score=False, random_state=None,
    verbose=0, warm_start=False
)
random_forest_model.fit(X_train, y_train)
titanic_predictions = random_forest_model.predict(X_test)
accuracy = random_forest_model.score(X_train, y_train)
important_features = (pd.Series(random_forest_model.feature_importances_, X_train.columns)
                      .sort_values(ascending=False))
submission = pd.DataFrame({'PassengerId': validation_data['PassengerId'], 'Survived': titanic_predictions})
submission.to_csv('data/titanic/submission-3rd.csv', index=False)















































