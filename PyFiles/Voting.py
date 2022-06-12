from sklearn.ensemble import VotingClassifier
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
X_train_split= pd.read_csv("x_train.csv")
y_train_split= pd.read_csv("y_train.csv")

X_val_split= pd.read_csv("x_validation.csv")
y_val_split= pd.read_csv("y_validation.csv")

lgbm_clf1 = joblib.load('LightGBM.joblib')
catboost = joblib.load("CatBoost.joblib")
voting_clf = VotingClassifier(
    estimators=[('lg', lgbm_clf1), ('cat', catboost)],
    voting='soft')
voting_clf.fit(X_train_split, y_train_split)
voting_pred=voting_clf.predict_proba(X_val_split)[:, 1]
print(roc_auc_score(y_val_split,voting_pred))
filename = "VotingClassifier.joblib"
joblib.dump(voting_clf, filename)