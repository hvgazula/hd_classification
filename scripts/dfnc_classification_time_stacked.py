#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:12:20 2019

@author: Harshvardhan
"""

import glob
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
from feast import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

sns.set_style("whitegrid")
cwd = os.path.dirname(os.getcwd())
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

subject_files = glob.glob(os.path.join(data_dir, 'hd_dfnc', 'rest_dfnc_sub_*_sess_001_results.mat'))
    
dfnc_3d = np.stack([sio.loadmat(subject)['FNCdyn'] for subject in sorted(subject_files)], axis=0)
    
# Fitting classifiers for each time point
demographics = pd.read_excel(os.path.join(data_dir, '20160420_vcalhoun_rest_demography_cag_info_new.xls'))

#algos = [BetaGamma, DISR, CIFE, CMIM, CondMI, Condred]
algos = [DISR]
#num_features_list = range(10, 51, 10)
num_features = 100
rbf = SVC(gamma='scale')
#classifiers = [lr, svc, rbf, rf, knn]
classifier = rbf
classifiers_string = ['rbf']

y_pred = []
for i in range(dfnc_3d.shape[1]):
    X = dfnc_3d[:, i, :]
    y = demographics.cap_d_group_id2.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y)

    select_feature_index = DISR(X_train, y_train, num_features)

    X_train_select = X_train[:, np.array(select_feature_index).astype(int)]
    y_train_select = y_train

    X_test_select = X_test[:, np.array(select_feature_index).astype(int)]
    y_test_select = y_test

    classifier.fit(X_train_select, y_train_select)

    y_pred.append(classifier.predict(X_test_select))
    
    