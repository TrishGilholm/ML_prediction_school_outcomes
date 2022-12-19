# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:00:05 2022

@author: uqpgilho

l1 regularisation main outcome nested cross validation
"""

import pandas as pd 
#import data year 3
df_pca = pd.read_csv("test_9_last.csv")

#split into X,y,group_labels
y = df_pca["RandN_BMS"]
X = df_pca.drop("RandN_BMS", axis = 1)

#group =df_pca["patient_id"]

y.value_counts()
#scale variables
from sklearn.preprocessing import StandardScaler
numeric_features = [ "SBPA", "PO2A", "FIO2A", "BEA", "ICU_HRS", "INTUB_HRS",
                    "PIM2_anz11_DEATH", "age_in_months", "Decile", "n_prev_admissions", "ICU_cumulative",
                    "INTUB_cumulative",
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
X_catgorical_dummy = pd.get_dummies(X_cat[categorical_features])
X = pd.concat([X_numeric_cols.reset_index(drop=True), X_catgorical_dummy.reset_index(drop=True)], axis=1)


X = X.drop(["IND_STATUS_0", "RETRIEV_0",
            "RS_HR124_0", "PUPILS_0",
            "ELECTIVE_0", "RECOVERY_0",
            "BYPASS_0", "CVVH_0", "PD_0",
            "PF_0", "HFO_0", "INO_0",
            "ECMO_0", "RACHS_risk_0",
            "INTUB_YN_0", "HD_0"], axis = 1)

list(X.columns)
#attempt at nested stratified grouped cross validation
from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score,\
    precision_score,recall_score, f1_score, accuracy_score, balanced_accuracy_score

#df_pca["stratify_vars"] =df_pca["RandN_BMS"].apply(str) + "_" + df_pca["ses_cat"].apply(str) + "_" + df_pca["year_level"].apply(str)
df_pca["stratify_vars"] =df_pca["RandN_BMS"].apply(str) + "_" + df_pca["ses_cat"].apply(str)
df_pca["stratify_vars"] = pd.Categorical(df_pca["stratify_vars"])
#set up grid

# Maximum tree leaves for base learners
num_leaves = [15, 31, 50]

# Boosting learning rate
learning_rate = [0.1, 0.3, 0.5]

# Number of boosted trees to fit
n_estimators = [50, 100, 200]

# L1 regularisation term on weights
reg_alpha = [0, 1e-03, 1e-02]

# L2 regularisation term on weights
reg_lambda = [0, 1e-03, 1e-02]

random_grid = {'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'n_estimators': n_estimators}


#define model
clf = LGBMClassifier(objective = "binary", is_unbalance = True )

#confgure cross validation procedure
cv_outer = StratifiedKFold(n_splits = 5)
#enumerate splits
fold_no = 1
outer_results_auc_GBM = list()
outer_results_auprc_GBM = list()
for train_ix, test_ix in cv_outer.split(X,df_pca.stratify_vars ):
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
    train.to_csv('train_fold_nextNAP_GBM_primary_9EE_' + str(fold_no) + '.csv')
    test.to_csv('test_fold_nextNAP_GBM_primary_9EE_' + str(fold_no) + '.csv')
    fold_no += 1
    #configure inner cross validation procedure
    cv_inner = StratifiedKFold(n_splits = 5)
    
    #define search
    search = GridSearchCV(clf, random_grid, scoring='roc_auc', cv=cv_inner.split(X_train_scaled, train.stratify_vars), refit=True, n_jobs = -1)
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
    outer_results_auc_GBM.append(roc_auc)
    outer_results_auprc_GBM.append(auprc)
    # report progress
    print('>auc=%.3f, est=%.3f, cfg=%s' % (roc_auc, result.best_score_, result.best_params_))

import pickle
with open("auc_GBM_nextNAP_last_v3", "wb") as fp:
    pickle.dump(outer_results_auc_GBM, fp)
    
with open("auc_GBM_nextNAP_last_v2", "rb") as fp:
    b= pickle.load(fp)

import pickle
with open("auprc_GBM_nextNAP_last_v3", "wb") as fp:
    pickle.dump(outer_results_auprc_GBM, fp)
    
with open("auprc_GBM_nextNAP_last_v2", "rb") as fp:
    outer_results_auprc_GBM= pickle.load(fp)     
    

mean(outer_results_auc_GBM), std(outer_results_auc_GBM) 
mean(outer_results_auprc_GBM), std(outer_results_auprc_GBM)
rel_auprc = [x/0.042 for x in outer_results_auprc_GBM]
mean(rel_auprc), std(rel_auprc)
    
    

#process each fold
import pandas as pd 
#import data for best split year 3 last admission
train_lastadmit = pd.read_csv("train_fold_nextNAP_GBM_primary_v3_5.csv")
test_lastadmit = pd.read_csv("test_fold_nextNAP_GBM_primary_v3_5.csv")

#subset test to subgroup
#test_lastadmit = test_lastadmit.loc[test_lastadmit['any_comorb'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['no_comorb'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['prematurity'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['heart'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['congenital'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['oncologic'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['resp'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['neuro'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['resp_inf'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['pneumo'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['Bronchiolitis'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['inv_inf'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['septic_shock'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['Asthma'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['diabetes'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['trauma'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['head_trauma'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['ELECTIVE'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['ELECTIVE'] == 0]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['admit_before_2008'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['admit_after_2008'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['QCH'] == 1]
#test_lastadmit = test_lastadmit.loc[test_lastadmit['QCH'] == 0]

y_train = train_lastadmit["RandN_BMS"]
X_train = train_lastadmit.drop("RandN_BMS", axis = 1)  

y_test = test_lastadmit["RandN_BMS"]
X_test = test_lastadmit.drop("RandN_BMS", axis = 1)  

numeric_features = [ "SBPA", "PO2A", "FIO2A", "BEA", "ICU_HRS", "INTUB_HRS",
                    "PIM2_anz11_DEATH", "n_prev_admissions", "age_in_months", "Decile", "ICU_cumulative",
                     "INTUB_cumulative", "TC1", "TC2", "TC3", "TC4", "TC5", "TC6",
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



#dummies train
categorical_features = ["GENDER", "IND_STATUS", "IADM_SC", "HADM_SC", "RETRIEV", "RS_HR124", "PUPILS",
                        "ELECTIVE", "RECOVERY", "BYPASS", "CVVH", "PD", "PF", "HFO", "INO", "ECMO",
                        "new_RA", "RACHS_risk", "INTUB_YN", "repeat_test",
                        "ses_cat", "HD"]



X_numeric_cols = X_train[numeric_features]
X_cat = X_train[categorical_features].astype("category")
X_catgorical_dummy = pd.get_dummies(X_cat[categorical_features])
X_train = pd.concat([X_numeric_cols.reset_index(drop=True), X_catgorical_dummy.reset_index(drop=True)], axis=1)


X_train = X_train.drop(["IND_STATUS_0", "RETRIEV_0",
            "RS_HR124_0", "PUPILS_0",
            "ELECTIVE_0", "RECOVERY_0",
            "BYPASS_0", "CVVH_0", "PD_0",
            "PF_0", "HFO_0", "INO_0",
            "ECMO_0", "RACHS_risk_0",
            "INTUB_YN_0",
             "HD_0"], axis = 1)

X_numeric_cols = X_test[numeric_features]
X_cat = X_test[categorical_features].astype("category")
X_catgorical_dummy = pd.get_dummies(X_cat[categorical_features])
X_test = pd.concat([X_numeric_cols.reset_index(drop=True), X_catgorical_dummy.reset_index(drop=True)], axis=1)


X_test = X_test.drop(["IND_STATUS_0", "RETRIEV_0",
            "RS_HR124_0", "PUPILS_0", "RECOVERY_0",
            "BYPASS_0", "CVVH_0", "PD_0", "ELECTIVE_0",
            "PF_0", "HFO_0", "INO_0",
            "ECMO_0", "RACHS_risk_0",
            "INTUB_YN_0",
             "HD_0"], axis = 1)


list(X_train.columns)
list(X_test.columns)


#X_test['HD_1'] = 0
#X_test['RACHS_risk_5'] = 0
#X_test['RACHS_risk_4'] = 0
#X_test['RACHS_risk_1'] = 0
#X_test['RACHS_risk_2'] = 0
#X_test['RACHS_risk_3'] = 0
#X_test['ECMO_1'] = 0
#X_test['PUPILS_1']=0
#X_test['INO_1']=0
#X_test['HFO_1']=0
#X_test['PF_1'] = 0
#X_test['PD_1'] = 0
#X_test['CVVH_1'] = 0
#X_test['HADM_SC_6']=0
#X_test['HADM_SC_3']=0
#X_test['HADM_SC_4']=0
#X_test['repeat_test_True']=0
#X_test['new_RA_VR']=0
#X_test['new_RA_R']=0
#X_test['IADM_SC_4']=0
#X_test['IADM_SC_1']=0
#X_test['IADM_SC_3']=0
#X_test['BYPASS_1']=0
#X_test['RETRIEV_1']=0
#X_test['RS_HR124_1']=0
#X_test['RECOVERY_1']=0
##X_test['INTUB_YN_1']=0
#X_test['IND_STATUS_1']=0
#X_test['ELECTIVE_1']=0

cols = X_train.columns.tolist()
X_test = X_test[cols]



from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

X_numeric_scale_train = pd.DataFrame(data = scale.fit_transform(X_train[numeric_features]),
                               columns = numeric_features)
X_cat_train = X_train.drop(numeric_features, axis=1)
X_train = pd.concat([X_numeric_scale_train.reset_index(drop=True), X_cat_train.reset_index(drop=True)], axis=1)
#scale X_test
X_numeric_scale_test = pd.DataFrame(data = scale.transform(X_test[numeric_features]),
                               columns = numeric_features)
X_cat_test = X_test.drop(numeric_features, axis=1)
X_test = pd.concat([X_numeric_scale_test.reset_index(drop=True), X_cat_test.reset_index(drop=True)], axis=1)


from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score,\
    precision_score,recall_score, f1_score, accuracy_score, balanced_accuracy_score


#fit RF with optimal hyperparameters - evaluate on test set
clf_best = LGBMClassifier(objective = "binary", reg_lambda=0,
                     reg_alpha = 0, num_leaves = 15, n_estimators = 50,
                     learning_rate=0.1, is_unbalance=True)
clf_best.fit(X_train, y_train)

#evaluate predictions 
y_test_model= clf_best.predict_proba(X_test)
fp_rates, tp_rates, threshold = roc_curve(y_test, y_test_model[:,1])
roc_auc = auc(fp_rates, tp_rates)

precision, recall, thresholds = precision_recall_curve(y_test, y_test_model[:,1])
prc_auc = auc(recall,precision)
#evaluate sensitivity/specificity/ PLR/NLR
my_dict_fold5= {
    "fp_rates": fp_rates,
    "tp_rates": tp_rates,
    "thresholds": threshold
}

fold5_fp_tp = pd.DataFrame.from_dict(my_dict_fold5)
fold5_fp_tp.to_csv("roc_nextNAP_fold5_primary_v3.csv")

precision, recall, thresholds = precision_recall_curve(y_test, y_test_model[:,1])

pr_fold= {
    "precision": precision[:-1],
    "recall": recall[:-1],
    "thresholds": thresholds
}

fold_pr = pd.DataFrame.from_dict(pr_fold)
fold_pr.to_csv("pr_year9_fold5_primary.csv")


import shap
import matplotlib.pyplot as pyplot
import numpy as np


explainer = shap.TreeExplainer(clf_best, feature_names = X_test.columns)
shap_values = explainer.shap_values(X_test)
feature_names = X_test.columns
shap_df = pd.DataFrame(shap_values[1], columns=feature_names)

vals = np.abs(shap_df.values).mean(0)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                               columns=['col_name', 'feature_importance_vals'])
shap_importance['normalised_imp']=(shap_importance['feature_importance_vals']-shap_importance['feature_importance_vals'].min())/(shap_importance['feature_importance_vals'].max()-shap_importance['feature_importance_vals'].min())
#shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

# rank: 
#shap_importance.set_index('col_name').rank()


shap_importance.to_csv("shap_importance_nextNAP_primary_fold5_v3.csv")




shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, plot_size = (18,13))
pyplot.savefig("SHAP_gbm_bar_year3_nestedcv_best.png", dpi=300)

shap.summary_plot(shap_values[1], X_test, show = False, max_display=20)
pyplot.gcf().axes[-1].set_aspect(100)
pyplot.gcf().axes[-1].set_box_aspect(100)
pyplot.savefig("SHAP_Year3_gbm_beeswarm_nestedcv_best.png", dpi=300)



