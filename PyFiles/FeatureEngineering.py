###Load data

import os
import pandas as pd
import gc
import numpy as np
import seaborn as sb
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


def load(data_dir: str) -> pd.DataFrame:
    identities = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    transactions = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    x_train = transactions.merge(identities, how='left', on='TransactionID')
    y_train = transactions.isFraud.values
    identities_test = pd.read_csv(os.path.join(data_dir, "test_identity.csv"))
    transactions_test = pd.read_csv(os.path.join(data_dir, "test_transaction.csv"))
    x_test = transactions_test.merge(identities_test, how='left', on='TransactionID')
    del transactions
    del identities
    del transactions_test
    del identities_test
    gc.collect()
    x_train.drop(['TransactionID', 'isFraud'], axis=1, inplace=True)
    x_test.drop(['TransactionID'], axis=1, inplace=True)
    return x_train, y_train, x_test


x_train, y_train, x_test = load(data_dir="data")

print("loading done")
x_test.columns = [f.replace("id-", "id_") for f in x_test.columns]

# removing nan values




def remove_nan(df_train):
    col_na = x_train.isna().sum()
    to_drop = col_na[(col_na / x_train.shape[0]) > 0.9].index
    use_col = [f for f in x_train.columns if f not in to_drop]
    return x_train[use_col]


x_train = remove_nan(x_train)
print("nan over 90% removed- done")
# TransactionDT modification
###first value is  86400= 60* 60 *24=number of seconds in a day
# data span= 6 months, max values=15811131 ->day=183
pd.options.mode.chained_assignment = None
x_train['TransactionDT'].min(), x_train['TransactionDT'].max()


def add_time_features(df):
    df['Transaction_day_of_week'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7)
    df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
    return df


x_train = add_time_features(x_train)



# add amount features
def add_amount_features(df):
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform(
        'mean')
    return df


x_train = add_amount_features(x_train)


def add_email_features(df):
    df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df['P_emaildomain'].str.split('.', expand=True)
    df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df['R_emaildomain'].str.split('.', expand=True)
    return df


x_train = add_email_features(x_train)
print("features addes")


# Replace nan values

def replace_nan(df_train):
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = [f for f in x_train.columns if (f not in categorical_features)]
    df_train[numerical_features] = df_train[numerical_features].fillna(df_train[numerical_features].mean())
    df_train[categorical_features] = df_train[categorical_features].fillna("missing")
    return df_train


x_train = replace_nan(x_train)
print("nan values removed")
# remove correlated features
corr = x_train.corr()
cor_matrix = corr.abs()






def drop_cor_features(df_train):
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    df_train = df_train.drop(to_drop, axis=1)
    return df_train


x_train = drop_cor_features(x_train)
print("correlated features droped")



# encoding strings
def encode_string(df_train):
    for col in df_train.columns:
        if df_train[col].dtype == "object":
            le = LabelEncoder()
            le.fit(list(df_train[col].astype(str).values))
            df_train[col] = le.transform(list(df_train[col].astype(str).values))
    return df_train


x_train = encode_string(x_train)
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder_one(data, feature):
    oh = OneHotEncoder()

    oh_df = pd.DataFrame(oh.fit_transform(data[[feature]]).toarray())
    oh_df.columns = oh.get_feature_names()

    for col in oh_df.columns:
        oh_df.rename({col: f'{feature}_' + col.split('_')[1]}, axis=1, inplace=True)

    new_data = pd.concat([data, oh_df], axis=1)
    new_data.drop(feature, axis=1, inplace=True)

    return new_data

#x_train= pd.get_dummies(x_train)
print("strings encoded")

# reduce memory usage





def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


x_train = downcast_dtypes(x_train)


print("datatypes downcasted")

x_train.to_csv('x_features_done.csv', index=False)

pd.DataFrame(y_train).to_csv("y_features_done.csv", index=False)
