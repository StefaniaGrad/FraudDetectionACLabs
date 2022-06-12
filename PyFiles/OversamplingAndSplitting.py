import os
import pandas as pd
import gc
import  numpy as np
import seaborn as sb
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch

def load() -> pd.DataFrame:
    x_train= pd.read_csv("x_features_done.csv")
    y_train= pd.read_csv("y_features_done.csv")
    return x_train, y_train
x_train, y_train=load()
y_train=y_train['0'].to_numpy()
g = sns.countplot(y_train)
g.set_xticklabels([0,1])
from imblearn.over_sampling import SMOTE
#Synthetic Minority Oversampling Technique
over = SMOTE(sampling_strategy=0.1)
X_train, Y_train =over.fit_resample(x_train, y_train)


X_train_split, X_testandval_split, y_train_split, y_testandval_split = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=7)
print(len(X_train_split), 'train examples')
print(len(X_testandval_split), 'validation and test examples')
X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(
    X_testandval_split, y_testandval_split, test_size=0.5, random_state=7)
print(len(X_val_split), 'validation examples')
print(len(X_test_split), 'test examples')

X_train_split.to_csv('x_train.csv', index=False)
pd.DataFrame(y_train_split).to_csv("y_train.csv", index=False)

X_val_split.to_csv('x_validation.csv', index=False)
pd.DataFrame(y_val_split).to_csv("y_validation.csv", index=False)

X_test_split.to_csv('x_test.csv', index=False)
pd.DataFrame(y_test_split).to_csv("y_test.csv", index=False)


