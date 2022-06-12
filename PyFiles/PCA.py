from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd
pca = PCA(n_components=0.90)

X_train = pd.read_csv("x_features_done.csv")
y_train = pd.read_csv("y_features_done.csv")
y_train = y_train['0'].to_numpy()

X_red = pca.fit_transform(X_train)
imp_col_no_after_pca = np.argmax(pca.components_)
imp_col_after_pca = X_train.columns[imp_col_no_after_pca]

print("The feature with most variance after PCA {}".format(imp_col_after_pca))
X_train_red, X_testandval_red, y_train_red, y_testandval_red = train_test_split(X_red, y_train, test_size=0.2)
X_val_red, X_test_red, y_val_red, y_test_red = train_test_split(X_testandval_red, y_testandval_red, test_size=0.5)
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train_red, y_train_red)
y_pred_xgb = xgb_clf.predict_proba(X_val_red)[:, 1]
print("Score obtained after PCA with XGBClassifier")
print(roc_auc_score(y_val_red , y_pred_xgb))
filename = "XGBClassifierWithPCA.joblib"
joblib.dump(xgb_clf, filename)

catboost2 = CatBoostClassifier()
catboost2.fit(X_train_red, y_train_red)
y_pred_catboost2=catboost2.predict_proba(X_val_red)[:, 1]
print("Score obtained after PCA with Catboot" )
print(roc_auc_score(y_val_red , y_pred_catboost2))

filename = "CatBoostWithPCA.joblib"
joblib.dump(catboost2, filename)
