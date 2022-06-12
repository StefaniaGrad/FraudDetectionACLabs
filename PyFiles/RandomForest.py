import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

X_train_split= pd.read_csv("x_train.csv")
y_train_split= pd.read_csv("y_train.csv")
y_train_split=y_train_split['0'].to_numpy()

X_val_split= pd.read_csv("x_validation.csv")
y_val_split= pd.read_csv("y_validation.csv")
y_val_split=y_val_split['0'].to_numpy()


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]

max_features = [20,30,40]

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [100, 200, 300]

min_samples_leaf = [100, 200, 4000]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf = RandomForestClassifier(random_state = 42)

rf_random =RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='roc_auc_score', n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_split, y_train_split)
print(rf_random.best_params_)



rnd_clf = RandomForestClassifier(
    max_depth=45, max_features=30, n_estimators=500, n_jobs=-1, min_samples_leaf=200
)
rnd_clf.fit(X_train_split, y_train_split)
y_pred_rnd = rnd_clf.predict_proba(X_val_split)[:, 1]
print(roc_auc_score(y_val_split,y_pred_rnd))

