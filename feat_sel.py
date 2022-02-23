#!/usr/bin/python3
# coding: utf-8

# Gene function prediction
# Miguel Romero, nov 3rd

import os
import sys
import datetime
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from time import time

# Own Libraries
from plots import *

# Metrics
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# Cross-validation and scaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Over-sampling and classifier Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

import shap
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',category=DeprecationWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)



def training_smote(X, y, term, n_splits, seed, plot):

  clf = xgb.XGBClassifier(booster='gbtree', n_jobs=n_jobs_clf, random_state=seed,
                          eval_metric="aucpr") #, use_label_encoder=False)
  rand_xgb = RandomizedSearchCV(clf, param_grid, scoring="recall",
                                n_jobs=n_jobs_cv, n_iter=n_iter, random_state=seed)

  sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  for train_index, test_index in sss.split(X, y):
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    Xtrain, Xtest = Xtrain.values, Xtest.values
    ytrain, ytest = ytrain.values, ytest.values

    if np.sum(ytest) > 0 and np.sum(ytest) > len(ytest) and \
      np.sum(ytrain) > 0 or np.sum(ytrain) > len(ytrain): break

  try:
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb)
    model = pipeline.fit(Xtrain, ytrain)
  except:
    pipeline = imbalanced_make_pipeline(RandomOverSampler(sampling_strategy='minority'), rand_xgb)
    model = pipeline.fit(Xtrain, ytrain)

  best_est = rand_xgb.best_estimator_
  explainer = shap.Explainer(best_est)
  shap_values = explainer(Xtrain)

  vals = shap_values.abs.values.mean(0)
  feature_importance = pd.DataFrame(list(zip(X.columns,vals)),columns=['col','val'])
  feature_importance.sort_values(by=['val'],ascending=False,inplace=True)

  if plot:
    shap.plots.beeswarm(shap_values, show=False, max_display=22)
    plt.title(term)
    plt.tight_layout()
    plt.savefig('{0}/{1}'.format(figs_path, term.replace(':','')), format='pdf', dpi=300)
    plt.close()

    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title(term)
    plt.tight_layout()
    plt.savefig('{0}/{1}'.format(figs_path, 'b'+term.replace(':','')), format='pdf', dpi=300)
    plt.close()

  feature_importance = feature_importance[feature_importance.val>0]
  return feature_importance.to_dict('records')

##################################
##################################
start_time = time()
dt = datetime.datetime.today()

PATH = "/users/grupofinke/ramirez/xgb"
# PATH = "/home/miguel/projects/omics/maize_data/xgb"
figs_path = "{0}/feat_sel_all".format(PATH)

seed = None # 202104  # seed for random state
n_splits = 5 # number of folds
n_iter = 5   # n_iter for xgboost
cpu = multiprocessing.cpu_count()
n_jobs_cv = cpu // 2
n_jobs_clf = 2 # cpu // n_jobs_cv

param_grid = {
        'max_depth': [3, 6, 10],
        'min_child_weight': [0.5, 3.0, 5.0, 8.0],
        'eta': [0.01, 0.05, 0.2, 0.4],
        'subsample': [0.5, 0.7, 0.9, 1.0]}


###
# prediction
###

data = pd.read_csv('{0}/data.csv'.format(PATH), dtype='float')
data = data.drop(['Gene', 'diver'], axis=1)

nclusters = [str(x) for x in range(10,101,10)] + ['n{0}'.format(x) for x in range(10,101,10)]
c_terms = [[x for x in os.listdir('{0}/spectral_all/{1}'.format(PATH, nc)) if '.csv' in x] for nc in nclusters]

sterm_list = list()
for x in c_terms:
  sterm_list = np.union1d(sterm_list,x)
term_list = ['GO:{0}'.format(x[2:].split('.csv')[0]) for x in sterm_list]

feat_data = list()
for ridx, (sterm, term) in enumerate(zip(sterm_list, term_list)):

  print('### {0} of {1}, {2}'.format(ridx+1, len(term_list), term))

  X = data.drop([term], axis=1)
  X = data[['clust', 'deg', 'neigh_deg', 'betw', 'clos', 'eccec', 'pager', 'const', 'hubs', 'auths', 'coren']]
  for i, nc in enumerate(nclusters):
    if sterm in c_terms[i]:
      dfc = pd.read_csv('{0}/spectral_all/{1}/{2}'.format(PATH, nc, sterm))
      X[nc] = dfc.label
  y = data[term]

  N = 10
  tmp = dict()
  for i in range(N):
    tmp1 = training_smote(X, y, term, n_splits, seed, i==(N-1))
    for d in tmp1:
      if d['col'] not in tmp: tmp[d['col']] = 0
      tmp[d['col']] += d['val']

  tmp = pd.DataFrame(list(tmp.items()), columns=['col','val'])
  tmp.val = tmp.val / N
  tmp.sort_values(by=['val'],ascending=False,inplace=True)
  tmp.to_csv('{0}/feat_sel_all/{1}.csv'.format(PATH,term.replace(':','')), index=False)

  thr = tmp.val.sum() * 0.9
  tmp['cum']= tmp.val.cumsum()
  tmp = tmp[tmp.cum <= thr]
  feat_data.append([term] + tmp.col.tolist())


file = open("{0}/feat_sel_all/top_feat.txt".format(PATH), "w")
file.write('\n'.join([','.join(x) for x in feat_data]))

print("--- {0:.2f} seconds ---".format(time() - start_time))
