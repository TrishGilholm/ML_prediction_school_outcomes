# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:00:05 2022

@author: uqpgilho

l1 regularisation main outcome nested cross validation
"""

import pandas as pd 
#import data year 3
df_pca = pd.read_csv("year9_excl_worst.csv")

#check any columns that need to be removed
list(df_pca.columns)


#split into X,y,group_labels
y = df_pca["RandN_BMS"]
X = df_pca.drop("RandN_BMS", axis = 1)



y.value_counts()

#scale variables
from sklearn.preprocessing import StandardScaler
numeric_features = [ "SBPA", "PO2A", "FIO2A", "BEA", "ICU_HRS", "INTUB_HRS", "n_prev_admissions",
                    "INTUB_cumulative", "ICU_cumulative",
                    "PIM2_anz11_DEATH", "age_in_months", "Decile",
                      "TC1", "TC2", "TC3", "TC4", "TC5", "TC6",
                    "TC7","TC8", "TC9", "TC10", "TC11", "TC12", "TC13", "TC14", "TC15", "TC16", "TC17",
                    "TC18", "TC19", "TC20", "TC21", "TC22", "TC23", "TC24", "TC25", "TC26", "TC27",
                    "TC28", "TC29", "TC30", "TC31", "TC32", "TC33", "TC34", "TC35", "TC36", "TC37",
                    "TC38", "TC39", "TC40", "TC41", "TC42", "TC43", "TC44", "TC45", "TC46", "TC47",
                    "TC48", "TC49", "TC50", "TC51", "TC52", "TC53", "TC54", "TC55", "TC56", "TC57",
                    "TC58", "TC59", "TC60", "TC61", "TC62", "TC63", "TC64", "TC65", "TC66", "TC67",
                    "TC68", "TC69", "TC70", "TC71", "TC72", "TC73", "TC74", "TC75", "TC76", "TC77",
                    "TC78", "TC79", "TC80", "TC81", "TC82", "TC83", "TC84", "TC85", "TC86", "TC87",
                    "TC88", "TC89", "TC90", "TC91", "TC92", "TC93", "TC94", "TC95", "TC96", "TC97",
                    "TC98", "TC99", "TC100", "TC101", "TC102", "TC103", "TC104", "TC105", "TC106",
                    "TC107", "TC108", "TC109", "TC110", "TC111", "TC112", "TC113", "TC114", "TC115",
                    "TC116"]




categorical_features = ["GENDER", "IND_STATUS", "IADM_SC", "HADM_SC", "RETRIEV", "RS_HR124", "PUPILS",
                        "ELECTIVE", "RECOVERY", "BYPASS", "CVVH", "PD", "PF", "HFO", "INO", "ECMO",
                        "new_RA", "RACHS_risk", "INTUB_YN", "repeat_test",
                        "ses_cat", "HD"]

scale = StandardScaler()

X_numeric_cols = X[numeric_features]
X_cat = X[categorical_features].astype("category")
X_catgorical_dummy = pd.get_dummies(X_cat[categorical_features], drop_first = True)
X = pd.concat([X_numeric_cols.reset_index(drop=True), X_catgorical_dummy.reset_index(drop=True)], axis=1)


#attempt at nested stratified grouped cross validation
from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score,\
    precision_score,recall_score, f1_score, accuracy_score, balanced_accuracy_score



df_pca["stratify_vars"] =df_pca["RandN_BMS"].apply(str) + "_" + df_pca["ses_cat"].apply(str)

df_pca["stratify_vars"] = pd.Categorical(df_pca["stratify_vars"])
#LR loop
#confgure cross validation procedure
cv_outer = StratifiedKFold(n_splits = 5)
#define model
clf = LogisticRegression(penalty = "l1",max_iter=10000, class_weight='balanced', solver='saga')
 #define search space
param_grid = {'C': [10 ** i for i in range(-7, 3)]}
#enumerate splits
outer_results_auc_LR = list()
outer_results_auprc_LR = list()
fold_no = 1
for train_ix, test_ix in cv_outer.split(X, df_pca.stratify_vars):
    print('\nouter fold %d\n' % fold_no)
    #split data
    X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
    y_train, y_test = y[train_ix], y[test_ix]
    
    #scale X_train
    X_numeric_scale_train = pd.DataFrame(data = scale.fit_transform(X_train[numeric_features]),
                                   columns = numeric_features)
    X_cat_train = X_train.drop(numeric_features, axis=1)
    X_train_scaled = pd.concat([X_numeric_scale_train.reset_index(drop=True), X_cat_train.reset_index(drop=True)], axis=1)
    #scale X_test
    X_numeric_scale_test = pd.DataFrame(data = scale.transform(X_test[numeric_features]),
                                   columns = numeric_features)
    X_cat_test = X_test.drop(numeric_features, axis=1)
    X_test_scaled = pd.concat([X_numeric_scale_test.reset_index(drop=True), X_cat_test.reset_index(drop=True)], axis=1)
    
    #save outer splits to csv
    train = df_pca.loc[train_ix,:]
    test = df_pca.loc[test_ix,:]
    train.to_csv('train_fold_year9_LR_primary_exclAW_worst_' + str(fold_no) + '.csv')
    test.to_csv('test_fold_year9_LR_primary_exclAW_worst_' + str(fold_no) + '.csv')
    fold_no += 1
    #configure inner cross validation procedure
    cv_inner = StratifiedKFold(n_splits = 5)
     #define search
    search = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=cv_inner.split(X_train_scaled, train.stratify_vars), refit=True, n_jobs = -1)
    #execute search
    result = search.fit(X_train_scaled, y_train)
    #get best performing model fit on the whole training set
    best_model = result.best_estimator_
    #evaluate model on the hold out data_set
    yprob = best_model.predict_proba(X_test_scaled)
    #evaluate model
    fp_rates, tp_rates, _ = roc_curve(y_test, yprob[:,1])
    roc_auc = auc(fp_rates, tp_rates)
    precision, recall, threshold = precision_recall_curve(y_test, yprob[:,1], pos_label=1)
    auprc = auc(recall, precision)
    #store result
    outer_results_auc_LR.append(roc_auc)
    outer_results_auprc_LR.append(auprc)
    # report progress
    print('>auc=%.3f, est=%.3f, cfg=%s' % (roc_auc, result.best_score_, result.best_params_))
    
import pickle
with open("auc_LR_year9_worst_exclAW", "wb") as fp:
    pickle.dump(outer_results_auc_LR, fp)
    
with open("auc_LR_year9_worst_exclAW", "rb") as fp:
    b= pickle.load(fp)

import pickle
with open("auprc_LR_year9_worst_exclAW", "wb") as fp:
    pickle.dump(outer_results_auprc_LR, fp)
    
with open("auprc_LR_year9_worst_exclAW", "rb") as fp:
    b= pickle.load(fp)     
    

mean(outer_results_auc_LR), std(outer_results_auc_LR) 
mean(outer_results_auprc_LR), std(outer_results_auprc_LR)
rel_auprc = [x/0.166 for x in outer_results_auprc_LR]
mean(rel_auprc), std(rel_auprc)




    