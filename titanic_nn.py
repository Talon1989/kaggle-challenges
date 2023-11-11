import numpy as np
import pandas as pd
from pygame.sprite import collide_circle_ratio
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utilities import one_hot_transformation
from tqdm import tqdm


'''
https://www.kaggle.com/competitions/titanic
 predict who will survive and who will die
'''


data = pd.read_csv('data/titanic/train.csv')
validation_data = pd.read_csv('data/titanic/test.csv')
columns = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
basic_columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch"]
N_EPOCHS = 200


filtered_data = data[basic_columns].dropna()
filtered_data['Sex'] = LabelEncoder().fit_transform(filtered_data['Sex'])
# filtered_data['Embarked'] = LabelEncoder().fit_transform(filtered_data['Embarked'])
# fares = filtered_data['Fare'].to_numpy().reshape([-1, 1])
# normalized_fares = StandardScaler().fit_transform(fares)
# filtered_data['Fare'] = normalized_fares
X_ = filtered_data.drop('Survived', axis=1).to_numpy()
y_ = one_hot_transformation(filtered_data['Survived'].to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=4/5, stratify=y_)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float64),
    torch.tensor(y_train, dtype=torch.float64))
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)


class NN(nn.Module):
    def __init__(self, in_shape, out_shape, dtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(d=dtype)
        self.layers = nn.Sequential(
            nn.Linear(in_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.layers(x)
        return output


# x_batch, y_batch = next(iter(dataloader))
model = NN(in_shape=X_.shape[1], out_shape=y_.shape[1])
optimizer = torch.optim.Adam(params=model.parameters(), lr=1/2_000)
criterion = nn.MSELoss()


with tqdm(total=N_EPOCHS) as pbar:
    for epoch in range(1, N_EPOCHS+1):
        losses = []
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            model.train()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())
        pbar.set_description('Epoch %d | Loss %.3f' % (epoch, np.mean(losses)))
        pbar.update(1)
        # print('Epoch %d | Loss %.3f' % (epoch, np.mean(losses)))


test_predictions = model(torch.tensor(X_test, dtype=torch.float64)).detach().numpy()
test_predictions = np.argmax(test_predictions, axis=1)
y_values = np.argmax(y_test, axis=1)
print('Accuracy on test data: %.3f' % ((np.sum(test_predictions == y_values)) / len(y_values)))


# to use to replace element of validation data with nan in Fare
val_data = validation_data[basic_columns[1:]]
# fare_nan_mask = validation_data['Fare'].isna()
# print('Rows with nan values in Fare:\n%s' % validation_data[fare_nan_mask])
# fare_mean = np.mean(validation_data['Fare'])
# validation_data['Fare'].fillna(fare_mean, inplace=True)
# print('Rows with old nan values in Fare:\n%s' % validation_data[fare_nan_mask])
val_data.loc[:, 'Sex'] = LabelEncoder().fit_transform(validation_data['Sex'])
validation = torch.tensor(val_data.to_numpy(), dtype=torch.float64)
predictions = torch.argmax(model(validation), dim=1).detach().numpy()
outputter = pd.DataFrame({'PassengerId': validation_data.PassengerId, 'Survived': predictions})
outputter.to_csv('data/titanic/submission-nn.csv', index=False)






























































































































































