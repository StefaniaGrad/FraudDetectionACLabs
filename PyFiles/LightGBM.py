import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import joblib

X_train_split= pd.read_csv("x_train.csv")
y_train_split= pd.read_csv("y_train.csv")
y_train_split=y_train_split['0'].to_numpy()

X_val_split= pd.read_csv("x_validation.csv")
y_val_split= pd.read_csv("y_validation.csv")
y_val_split=y_val_split['0'].to_numpy()

light_gbm = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
light_gbm.fit(X_train_split, y_train_split)
y_pred_lgbm = light_gbm.predict_proba(X_val_split)[:,1]
print("Score obtained with LightGBM: ")
print(roc_auc_score(y_val_split,y_pred_lgbm))
filename = "LightGBM.joblib"
joblib.dump(light_gbm, filename)