import numpy as np
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import joblib

import torch

from sklearn.metrics import roc_auc_score

X_train_split= pd.read_csv("x_train.csv")
y_train_split= pd.read_csv("y_train.csv")
y_train_split=y_train_split['0'].to_numpy()

X_val_split= pd.read_csv("x_validation.csv")
y_val_split= pd.read_csv("y_validation.csv")
y_val_split=y_val_split['0'].to_numpy()

catboost =  CatBoostClassifier()
print(X_train_split)
print(y_train_split)
catboost.fit(X_train_split,y_train_split)
y_pred_catboost=catboost.predict_proba(X_val_split)[:, 1]
print("Score obtained using catboot")
print(roc_auc_score(y_val_split,y_pred_catboost))
filename = "CatBoost.joblib"
joblib.dump(catboost, filename)
