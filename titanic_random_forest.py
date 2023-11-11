import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


'''
https://www.kaggle.com/competitions/titanic
 predict who will survive and who will die
'''


train_data = pd.read_csv('data/titanic/train.csv')
test_data = pd.read_csv('data/titanic/test.csv')


columns = train_data.columns.to_numpy()
relevant_features = ['Pclass', 'Sex', 'Young', 'SibSp', 'Parch', 'Embarked']


women = train_data[train_data.Sex == 'female']['Survived']
print('%% of women who survived: %.2f' % ((np.sum(women) / len(women)) * 100))

men = train_data[train_data.Sex == 'male']['Survived']
print('%% of men who survived: %.2f' % ((np.sum(men) / len(men)) * 100))


# BUILDING Y TARGET AND X WITH BASIC FEATURES
y = train_data['Survived']
basic_features = ["Pclass", "Sex", "SibSp", "Parch"]
# convert categorical column to dummy variables
basic_x_train = pd.get_dummies(train_data[basic_features]).dropna()
basic_x_test = pd.get_dummies(test_data[basic_features])


# RANDOM FOREST CLASSIFIER
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(basic_x_train, y)
preds = model.predict(basic_x_test)
output_1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': preds})


# BUILDING Y TARGET AND X
y = train_data['Survived']
train_data['Young'] = train_data['Age'] <= 35
train_data = train_data.drop('Age', axis=1)
test_data['Young'] = test_data['Age'] <= 35
test_data = test_data.drop('Age', axis=1)
# convert categorical column to dummy variables
x_train = pd.get_dummies(train_data[relevant_features])
x_train.drop('Sex_female', axis=1, inplace=True)
x_test = pd.get_dummies(test_data[relevant_features])
x_test.drop('Sex_female', axis=1, inplace=True)


# RANDOM FOREST CLASSIFIER
model = RandomForestClassifier(n_estimators=256, max_depth=8, random_state=42)
model.fit(x_train, y)
preds = model.predict(x_test)
output_2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': preds})
output_2.to_csv('data/titanic/submission-2.csv', index=False)































