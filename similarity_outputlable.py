# general and data handling
import numpy as np
import pandas as pd
import os
from collections import Counter
from IPython.display import display
import re
import openpyxl as xl
from scipy.spatial.distance import cdist
import copy

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
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from sklearn.model_selection import RandomizedSearchCV

from skopt import BayesSearchCV
import xgboost as xgb

# Graphing
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

import argparse

# import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(24)

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def data_split(df_remain, X_test, y_test, X_total_remain, y_total_remain, fp, random_state = 0, n_clusters = 85):
    # Get the MACCS fingerprints for remaining set
    fingerprints = copy.deepcopy(fp)
    
    # Convert fingerprints to a numpy array
    fp_matrix = np.array([list(fp) for fp in fingerprints])
    
    # Use KMeans clustering on the fingerprints
    kmeans = KMeans(n_clusters=n_clusters, random_state=4).fit(fp_matrix)
    df_remain['cluster'] = kmeans.labels_

    X_train = []
    X_val = []
    y_train = []
    y_val = []

    # For each cluster, do a train-validation split for the remaining set
    for i in range(n_clusters):
        cluster_indices = df_remain[df_remain['cluster'] == i].index
        X_train_cluster, X_val_cluster, y_train_cluster, y_val_cluster = train_test_split(X_total_remain[cluster_indices], y_total_remain[cluster_indices], test_size=0.1, random_state=random_state)

        X_train.extend(X_train_cluster)
        X_val.extend(X_val_cluster)
        y_train.extend(y_train_cluster)
        y_val.extend(y_val_cluster)
        
    return np.array(X_train), np.array(X_val), X_test, np.array(y_train), np.array(y_val), y_test

parser = argparse.ArgumentParser(description='chemical')
parser.add_argument('-split', type=str, default="similar", help='similar or dissimilar data set')
parser.add_argument('-subset', type=float, default=0.1, help='subset of weak data')
parser.add_argument('-iteration', type=int, default=25, help='number of iteration')
parser.add_argument('-threshold', type=float, default=0.9, help='similarity threshold')
parser.add_argument('-measure', type=str, default="maccs", help='similarity measure')
parser.add_argument('-seed', type=float, default=0.0, help='random seed for data split')

args = parser.parse_args()

if args.split == "similar":
    df_remain = pd.read_excel('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/similar test set.xlsx', sheet_name='remaining_set', engine='openpyxl')
    df_test = pd.read_excel('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/similar test set.xlsx', sheet_name='test_set', engine='openpyxl')
    best_params = {'colsample_bylevel': 0.5362386927724025, 'colsample_bynode': 0.7846605864264999, 'colsample_bytree': 0.5869205526328731,
                   'gamma': 0.120232287455155, 'learning_rate': 0.04247055522398333, 'max_delta_step': 1, 'max_depth': 49,
                   'min_child_weight': 1.407905723403665, 'n_estimators': 1247, 'reg_alpha': 0.4174734818527416,
                   'reg_lambda': 0.9800779693705134, 'scale_pos_weight': 4.374540085845599, 'subsample': 0.9287718676241973,
                   'booster': 'gbtree'}
    cluster_num = 70
elif args.split == "dissimilar":
    df_remain = pd.read_excel('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/remaining_set.xlsx',engine='openpyxl')
    df_test = pd.read_excel('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/test_set.xlsx',engine='openpyxl')
    best_params = {'colsample_bylevel': 0.6889841308028768, 'colsample_bynode': 0.6415107155640306, 'colsample_bytree': 0.6224045119598769,
                   'gamma': 0.17462902013639597, 'learning_rate': 0.04979875790770406, 'max_delta_step': 1, 'max_depth': 46,
                   'min_child_weight': 1.9095319307329182, 'n_estimators': 1119, 'reg_alpha': 0.11535043067842797,
                   'reg_lambda': 1.0902914563546875, 'scale_pos_weight': 1.824559842279296, 'subsample': 0.9872630630104621,
                   'booster': 'gbtree'}
    cluster_num = 85

df_remain['mol'] = [AllChem.MolFromSmiles(smiles) for smiles in df_remain['SMILES']]
df_remain_fp = [GetMACCSKeysFingerprint(mol) for mol in df_remain['mol']]
## Split FP to multiple columns so that they can be easily combined with others
fp_remain = pd.DataFrame(np.array(df_remain_fp))
## Combine with pH and T
df_remain_new = pd.concat([fp_remain, df_remain['pH'], df_remain['T']], axis=1)
X_total_remain = np.array(df_remain_new)
y_total_remain = np.array(df_remain['Log k'])

# Get the X_train_fp finger print
if args.measure == 'maccs':
    X_train_fp = copy.deepcopy(df_remain_fp)
elif args.measure == 'morg':
    X_train_fp = [GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(m),2, nBits = 2048,
                                            useChirality=False) for m in df_remain['SMILES'].values.tolist()]


df_test['mol'] = [AllChem.MolFromSmiles(smiles) for smiles in df_test['SMILES']]
df_test_fp = [GetMACCSKeysFingerprint(mol) for mol in df_test['mol']]
## Split FP to multiple columns so that they can be easily combined with others
fp_test = pd.DataFrame(np.array(df_test_fp))
## Combine with pH and T
df_test_new = pd.concat([fp_test, df_test['pH'], df_test['T']], axis=1)
X_test = np.array(df_test_new)
y_test = np.array(df_test['Log k'])

df_target = pd.read_csv('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/DSSTox (100%).csv')
df_target = df_target.sample(frac=args.subset, random_state=1, replace=True, ignore_index=True)
df_target['mol'] = [AllChem.MolFromSmiles(smiles) for smiles in df_target['SMILES']]
df_target_fp = [GetMACCSKeysFingerprint(mol) for mol in df_target['mol']]
## Split FP to multiple columns so that they can be easily combined with others
df_target_new = pd.DataFrame(np.array(df_target_fp))
## Combine with pH and T
df_target_new['pH'] = [7. / 14. for _ in df_target_new[0]]  # ph default to 7
df_target_new['T'] = [25. for _ in df_target_new[0]] # T default to 25
X_target = np.array(df_target_new)
hash_table = {tuple(X_target[i]): df_target['SMILES'][i] for i in range(X_target.shape[0])}
print(f"Successfully create hashtable with shape {len(hash_table)}")

# Get the X_target_fp finger print
if args.measure == 'maccs':
    X_target_fp = copy.deepcopy(df_target_fp)
elif args.measure == 'morg':
    X_target_fp = [GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(m),2, nBits = 2048,
                                            useChirality=False) for m in df_target['SMILES'].values.tolist()]
    
df_1_all = pd.read_excel('/home/drought/chemical/GenerateSimilarity/FinalVersion/data/GCM prediction 2.xlsx', sheet_name='0.5-0.4', engine='openpyxl')
df_1 = df_1_all.sample(n=200, random_state=0)
df_1['mol'] = [AllChem.MolFromSmiles(smiles) for smiles in df_1['SMILES']]

# Get finger print
X_gcm_fp = [GetMACCSKeysFingerprint(mol) for mol in df_1['mol']]

df_1['fp'] = [MACCSkeys.GenMACCSKeys(mol) for mol in df_1['mol']]
df_fp_1 = [MACCSkeys.GenMACCSKeys(mol) for mol in df_1['mol']]

# Convert each fingerprint to a numpy array, then stack them vertically
fp_1_array = np.vstack([np.array(fp) for fp in df_fp_1])
fp_1 = pd.DataFrame(fp_1_array)

# Combine with pH and T
df_new_1 = pd.concat([fp_1, df_1['pH'].reset_index(drop=True), df_1['T'].reset_index(drop=True)], axis=1)
display(df_new_1)
print(df_new_1.shape)
X_total_1 = np.array(df_new_1)
y_total_1 = np.array(df_1['Log k'])


X_train, X_val, X_test, y_train, y_val, y_test = data_split(df_remain, X_test, y_test, X_total_remain, y_total_remain, fp=df_remain_fp, random_state=int(args.seed), n_clusters=cluster_num)


def run_iterative_pseudo_labeling(hyperparameter, X_train, y_train, X_test, y_test, X_val, y_val, X_target,
                                  X_train_fp, X_target_fp, X_train_gcm, y_train_gcm, X_gcm_fp,
                                  num_iteration=25, simi_threshold=0.9):
    '''
    X_train, y_train, X_test, y_test, X_target -- np.array()
    X_train_fp -- list() : list of fingerprint
    X_target_fp -- list() : list of fingerprint
    similarity_vec -- np.array()
    index_target_dic -- dict(fp:int) : index of key fingerprint in target_fp list
                                 used for similarity_vec
    index_all_dic -- dict(fp:int) : index of key fingerprint in all_fp list
                              used for similarity_vec
    '''
    X_target_selected = X_train[[False] * len(X_train)]
    y_target_selected = y_train[[False] * len(y_train)]

    result_summary = pd.DataFrame(columns=['Qualified pseudo labels', 'MAE_test', 'RMSE_test', 'R2_test'])
    best_MAE_test, best_RMSE_test, best_R2_test = float('inf'), float('inf'), float('-inf')
    X_target_copy = copy.deepcopy(X_target)
    X_target_fp_copy = copy.deepcopy(X_target_fp)
    X_train_fp_copy = copy.deepcopy(X_train_fp)


    for iteration in range(num_iteration):
        X_concat = np.concatenate([X_train, X_target_selected], axis=0)
        y_concat = np.concatenate([y_train, y_target_selected], axis=0)
        cur_model = XGBRegressor(**hyperparameter)
        cur_model.fit(X_concat, y_concat, eval_set=[(X_val, y_val)], verbose=False)
        target_preds = cur_model.predict(X_target_copy)
        y_test_preds = cur_model.predict(X_test)

        MAE_test, RMSE_test, R2_test = calculate_metrics(y_test, y_test_preds)
        print('Iter {} Avg test MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}'.format(iteration+1, MAE_test, RMSE_test, R2_test))
        
        fp_similarity = []
        
        X_train_fp_copy = X_train_fp_copy + X_gcm_fp
        for i in range(len(X_target_fp_copy)):
            similarity = []
            for j in range(len(X_train_fp_copy)):
                temp = DataStructs.FingerprintSimilarity(X_target_fp_copy[i], X_train_fp_copy[j])
                similarity.append(temp)
            fp_similarity.append(max(similarity))
        fp_similarity = np.array(fp_similarity)

        # update target selected
        X_target_selected = np.concatenate([X_target_selected, X_target_copy[fp_similarity > simi_threshold]], axis=0)
        y_target_selected = np.concatenate([y_target_selected, target_preds[fp_similarity > simi_threshold]], axis=0)

        if iteration >= 0:
            selected = pd.DataFrame({'SMILES': [hash_table[tuple(X_target_selected[i])] for i in range(X_target_selected.shape[0])], 'pseudo label': y_target_selected})
            print(X_target_selected.shape)
            print(y_target_selected.shape)
            print(selected.shape)
            temp_path = f'/home/drought/chemical/GenerateSimilarity/FinalVersion/result11/{args.subset}_x_y_selected_at_iteration_{iteration}.csv'
            selected.to_csv(temp_path)
            print(f'Successfully store x y selected csv for iteration {iteration}')

        
        qualified_labels = len(y_target_selected)
        print('Qualified pseudo labels:', qualified_labels)
        # update target_fp and train_fp list
        # filter -- np.array()
        filter = fp_similarity > simi_threshold

        X_train_fp_copy = [i for indx,i in enumerate(X_target_fp_copy) if filter[indx] == True]
        X_target_fp_copy = [i for indx,i in enumerate(X_target_fp_copy) if filter[indx] == False]

        # update target
        X_target_copy = X_target_copy[fp_similarity <= simi_threshold]

        if R2_test > best_R2_test:
            best_MAE_test = MAE_test
            best_RMSE_test = RMSE_test
            best_R2_test = R2_test
        
        result = pd.DataFrame([[qualified_labels, MAE_test, RMSE_test, R2_test]], columns=result_summary.columns)
        result_summary = pd.concat([result_summary, result], ignore_index=True)



        
        if len(X_target_fp_copy) == 0 or len(X_train_fp_copy) == 0:
            best = pd.DataFrame([[0, best_MAE_test, best_RMSE_test, best_R2_test]], columns=result_summary.columns)
            result_summary = pd.concat([result_summary, best], ignore_index=True)
            return result_summary
    
    best = pd.DataFrame([[0, best_MAE_test, best_RMSE_test, best_R2_test]], columns=result_summary.columns)
    result_summary = pd.concat([result_summary, best], ignore_index=True)
    print(X_target_selected.shape)
    print(y_target_selected.shape)
    # selected = pd.DataFrame({'SMILES': [hash_table[tuple(X_target_selected[i])] for i in range(X_target_selected.shape[0])], 'pseudo label': y_target_selected})
    return result_summary

print("start pseudo labelling")
result_summary_out = run_iterative_pseudo_labeling(best_params, X_train, y_train, X_test, y_test, X_val, y_val, X_target,
                                                   X_train_fp, X_target_fp, X_train_gcm=X_total_1, y_train_gcm=y_total_1, X_gcm_fp=X_gcm_fp,
                                                   num_iteration=args.iteration, simi_threshold=args.threshold)

stored_path = f"./results/{args.split}_{args.subset}_{args.iteration}_{args.threshold}_{args.measure}_{args.seed}.csv"
result_summary_out.to_csv(stored_path)