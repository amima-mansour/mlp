import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold # import KFold

# read dataset
columns = []
for i in range(32):
    columns.append('column_' + str(i))
df = pd.read_csv('data.csv', names=columns)
# numerize column_1
df['column_1'] = df['column_1'].map({'M': 1, 'B': 0})
# split dataset
target = df.column_1
# create training and testing vars
#X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
#X_train = X_train.drop('column_1', axis=1)
#X_test = X_test.drop('column_1', axis=1)

X = df.values # create an array
y = df['column_1'].values # Create another array
kf = KFold(n_splits=3) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]