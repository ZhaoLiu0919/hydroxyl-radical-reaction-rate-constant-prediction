# %% [markdown]
# # Library import

# %%
# general and data handling
import numpy as np
import pandas as pd
import os
from collections import Counter
from IPython.display import display
import re
import openpyxl as xl
from scipy.spatial.distance import cdist

# Required RDKit modules
import rdkit as rd
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import RDConfig
from rdkit.Chem import PandasTools
from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

# modeling
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn import metrics, model_selection
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, cross_val_score, train_test_split, validation_curve

import sklearn.linear_model as skl_lm
from sklearn.metrics import classification_report, mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from sklearn.model_selection import RandomizedSearchCV

#from skopt import BayesSearchCV
#from bayes_opt import BayesianOptimization
import xgboost as xgb

# Graphing
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

import pickle

# %% [markdown]
# # Functions

# %%
def generate_predict_report(model, X, y):
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1)  # To make sure the format is array([1, 2, 3..]).
    MAE = mean_absolute_error(y, y_pred)
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    R2 = r2_score(y, y_pred)
    print(f'MAE: {MAE}')
    print(f'RMSE: {RMSE}')
    print(f'R2: {R2}')

    # Data scatter of predicted values
    plt.figure();plt.clf()
    plt.scatter(y, y_pred, marker='.', color='blue')
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("Prediction")
    plt.show()

    dict_test = {'MAE': MAE, 'RMSE': RMSE, 'R2': R2}
    return dict_test

def generate_train_report(model, X_train, y_train):
    print("-------------------------------------------------------------------------------------------")
    print(f"Train report for model {model}:")
    return generate_predict_report(model, X_train, y_train)

def generate_val_report(model, X_val, y_val):
    print("-------------------------------------------------------------------------------------------")
    print(f"Validation report for model {model}:")
    return generate_predict_report(model, X_val, y_val)
    
def generate_test_report(model, X_test, y_test):
    print("-------------------------------------------------------------------------------------------")
    print(f"Test report for model {model}:")
    return generate_predict_report(model, X_test, y_test)


def data_split(df, X_total, y_total, random_state, n_clusters):
    # Get the MACCS fingerprints
    fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi)) for smi in df['SMILES']]
    
    # Convert fingerprints to a numpy array
    fp_matrix = np.array([list(fp) for fp in fingerprints])
    
    # Use KMeans clustering on the fingerprints
    kmeans = KMeans(n_clusters=n_clusters, random_state=4).fit(fp_matrix)
    df['cluster'] = kmeans.labels_

    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []

    # For each cluster, do a train-validation-test split
    for i in range(n_clusters):
        cluster_indices = df[df['cluster'] == i].index
        X_remain, X_test_cluster, y_remain, y_test_cluster = train_test_split(X_total[cluster_indices], y_total[cluster_indices], test_size=0.1, random_state=random_state)
        X_train_cluster, X_val_cluster, y_train_cluster, y_val_cluster = train_test_split(X_remain, y_remain, test_size=0.1, random_state=random_state)

        X_train.extend(X_train_cluster)
        X_val.extend(X_val_cluster)
        X_test.extend(X_test_cluster)
        y_train.extend(y_train_cluster)
        y_val.extend(y_val_cluster)
        y_test.extend(y_test_cluster)
        
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    return train_metrics, val_metrics, test_metrics

# %% [markdown]
# # Data loading and processing

# %%
# Load data files
df = pd.read_excel('./Datasets.xlsx', sheet_name='Primary(pH+T)',engine='openpyxl')
df['mol'] = [AllChem.MolFromSmiles(smiles) for smiles in df['SMILES']]
df['fp'] = [GetMACCSKeysFingerprint(mol) for mol in df['mol']]
df_fp = [GetMACCSKeysFingerprint(mol) for mol in df['mol']]

## Split FP to multiple columns so that they can be easily combined with others
fp = pd.DataFrame(np.array(df_fp))

## Combine with pH and T
df_new = pd.concat([fp, df['pH'], df['T']], axis=1)
display(df_new)
print(df_new.shape)
X_total = np.array(df_new)
y_total = np.array(df['Log k'])

# %% [markdown]
# # Model optimization

# %%
def xgboost_eval(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, reg_lambda, scale_pos_weight, max_delta_step, learning_rate, X_train_list, y_train_list, X_val_list, y_val_list, random_states, booster):
    avg_score = 0
    
    for i, random_state in enumerate(random_states):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'max_delta_step': int(max_delta_step),
            'learning_rate': learning_rate,
            'booster': booster,
            'seed': int(random_state)
        }
        
        model = xgb.XGBRegressor(**params)
        
        model.fit(X_train_list[i], y_train_list[i])
        
        predictions = model.predict(X_val_list[i])
        
        # Taking the square root of the mean squared error to get the root mean squared error (RMSE)
        score = np.sqrt(mean_squared_error(y_val_list[i], predictions))
        avg_score += score
    
    # Return the negative average root mean squared error
    return -avg_score / len(random_states)

# %%
def hyperparameter_tuning_bayesian(df, X_total, y_total, random_states, n_clusters, n_iter=100, init_points=5):
    X_train_list = []
    X_val_list = []
    y_train_list = []
    y_val_list = []
    
    # Assuming data_split is a function defined elsewhere that splits the data
    for random_state in random_states:
        X_train, X_val, _, y_train, y_val, _ = data_split(df, X_total, y_total, random_state, n_clusters)
        
        # Adding the second dataset to the training set
        X_train = np.concatenate((X_train, X_total_1), axis=0)
        y_train = np.concatenate((y_train, y_total_1), axis=0)
        
        X_train_list.append(X_train)
        X_val_list.append(X_val)
        y_train_list.append(y_train) 
        y_val_list.append(y_val)
    
    def xgboost_optimization(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, reg_lambda, scale_pos_weight, max_delta_step, learning_rate, booster_index):
        booster = ['gbtree', 'dart'][int(booster_index)]
        return xgboost_eval(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, reg_lambda, scale_pos_weight, max_delta_step, learning_rate, X_train_list, y_train_list, X_val_list, y_val_list, random_states, booster)
    
    optimizer = BayesianOptimization(
        f=xgboost_optimization,
        pbounds={
            'n_estimators': (1000, 1500),
            'max_depth': (40, 50),
            'gamma': (0.1, 0.4),
            'min_child_weight': (1, 5),
            'subsample': (0.8, 1.0),
            'colsample_bytree': (0.5, 0.9),
            'colsample_bylevel': (0.5, 0.7),
            'colsample_bynode': (0.6, 0.8),
            'reg_alpha': (0.1, 0.5),
            'reg_lambda': (0.5, 1.5),
            'scale_pos_weight': (1.0, 5.0),
            'max_delta_step': (1, 3),
            'learning_rate': (0.02, 0.05),
            'booster_index': (0, 1)
        },
        random_state=42,
        verbose=2
    )


    optimizer.maximize(n_iter=n_iter, init_points=init_points)

    best_params = optimizer.max['params']

    # Convert the continuous booster_index to its corresponding category
    best_params['booster'] = ['gbtree', 'dart'][int(best_params.pop('booster_index'))]

    # Make sure some of the parameters are integers
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['max_delta_step'] = int(best_params['max_delta_step'])

    return best_params

# %%
random_states = [0, 1, 2, 3, 4]
best_params = hyperparameter_tuning_bayesian(df, X_total, y_total, random_states, n_clusters=70)
print("Best Hyperparameters:", best_params)

# %% [markdown]
# # Modeling

# %%
# Assign the best hyperparameters to a variable
best_params = {'colsample_bylevel': 0.5362386927724025, 'colsample_bynode': 0.7846605864264999, 'colsample_bytree': 0.5869205526328731, 'gamma': 0.120232287455155, 'learning_rate': 0.04247055522398333, 'max_delta_step': 1, 'max_depth': 49, 'min_child_weight': 1.407905723403665, 'n_estimators': 1247, 'reg_alpha': 0.4174734818527416, 'reg_lambda': 0.9800779693705134, 'scale_pos_weight': 4.374540085845599, 'subsample': 0.9287718676241973, 'booster': 'gbtree'}

# %%
result_summary = pd.DataFrame(columns=['Random State', 'MAE_train', 'RMSE_train', 'R2_train', 
                                       'MAE_val', 'RMSE_val', 'R2_val', 'MAE_test', 'RMSE_test', 'R2_test'])

for i in range(5):
    random_state = i
    n_clusters = 70  # Or any other number depending on your dataset
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(df, X_total, y_total, random_state, n_clusters)
    
    # Add the second dataset into training set
    X_train = np.concatenate((X_train), axis=0)
    y_train = np.concatenate((y_train), axis=0)
    
    model_optimized = xgb.XGBRegressor(**best_params)
    model_optimized.fit(X_train, y_train)
    train_report = generate_train_report(model_optimized, X_train, y_train)
    val_report = generate_val_report(model_optimized, X_val, y_val)
    test_report = generate_test_report(model_optimized, X_test, y_test)

    results = pd.Series([i] + list(train_report.values()) + list(val_report.values()) + list(test_report.values()), index=result_summary.columns)
    result_summary = result_summary.append(results, ignore_index=True)

print(result_summary)


