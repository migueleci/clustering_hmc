import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm, tnrange
from time import time

import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold

from evaluate import *

import shap

np.seterr(divide='ignore', invalid='ignore')


# read a txt file containing a list, one item per line
def readListFile(filename):
  file = open(filename, 'r')
  tmp = [x.strip() for x in file.readlines()]
  file.close()
  return np.array(tmp)


# pretty print of results for a model
def pprint(model, pooled, macro, macrow, time):
    print("## {0}:{1}pooled:\t{2:.4f}\t\tmacro:\t{3:.4f}\t\tmacrow:\t{4:.4f}\t\tTime:\t{5:.4f}".format(
      model,'\t' if len(model) > 7 else '\t\t',pooled, macro, macrow, time)
    )

# %%

root_list = readListFile("data/roots.txt")
results = pd.DataFrame(columns=["Root","Models","time",
                                "micro","macro","macrow"])
results_pl = pd.DataFrame(columns=["Root","Level",
                                   "micro","macro","macrow"])
cpu = multiprocessing.cpu_count()
seed = 220120
nlevels = 0

for j, root in enumerate(root_list):
  if j>0: print()
  print("{0}. Root: {1}".format(j+1, root))
  print("{0}------------------".format('--' if j>8 else '-'))

  """
    Read sub-hierarchy data:
     - structural properties of GCN
     - probabilities (ratio) of association for each function
     - true labels for each function
     - lcl order, info per level: ancestors and children
     - map of parents for each function
  """
  path = "data/{0}".format(root.replace(':',''))
  labels = pd.read_csv("{0}/labels.csv".format(path)) # labels - true gene-function associations
  labels = labels.drop([root], axis=1) # drop info of root, not used
  labels_order = labels.columns
  gene_list = pd.read_csv("{0}/genes.csv".format(path), names=["idx"])
  gene_list = gene_list.idx.tolist()

  lcl = open("{0}/lcl.txt".format(path), 'r') # load lcl order
  lcl = [x.strip() for x in lcl.readlines()]

  data = pd.DataFrame()
  for clust in next(os.walk('clustering'))[1]:
    clf = [x for x in os.listdir('clustering/{0}'.format(clust)) if 'GO:'+x.split('.')[0][2:] in labels_order]
    for file in clf:
      # if 'GO:'+file.split('.')[0][2:] in leaves: ###############################
      tmp = pd.read_csv('clustering/{0}/{1}'.format(clust,file))
      data['{0}_{1}'.format(clust,file.split('.')[0][2:])] = tmp.label
  data = data.loc[gene_list].reset_index(drop=True)

  # load both models and evaluation class
  estimator = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=seed) # Random forest classifier
  e = Evaluate()

  feature_importance = np.zeros(data.shape[1])
  N = 10
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  for train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="SHAP"): # set the same folds for each model

    """
      create y and x dataset for the current hierarchy
    """
    y = labels.copy() # labels for all functions in level (i.e., children)
    X = data.copy() # random forest dataset, the output of the previous level is different for both classifiers
    y_train, X_train = y.loc[train_index], X.loc[train_index] # train-test split of y

    """
    training and prediction for the current level. A special case arise for
    random forest when there is only ony child in the level, in that case the
    output of the rf is different
    """
    # random forest
    estimator.fit(X_train, y_train)
    explainer = shap.TreeExplainer(estimator)
    shap_values = np.array(explainer.shap_values(X_train))

    fimp = np.zeros(X.shape[1])
    for fidx in range(len(labels_order)):
      vals = abs(shap_values[fidx*2+1]).mean(0)
      fimp += vals

    fimp = fimp / len(labels_order)
    feature_importance += fimp

  fimp = pd.DataFrame(list(zip(X.columns, feature_importance)), columns=['col','val'])
  fimp.val = fimp.val / N
  fimp.sort_values(by=['val'],ascending=False,inplace=True)
  fimp = fimp[fimp.val > 0]
  fimp.to_csv('{0}/features.csv'.format(path), index=False)

  thr = fimp.val.sum() * 0.9
  fimp['cum']= fimp.val.cumsum()
  fimp = fimp[fimp.cum <= thr]



  # array for results
  pooled, etime = list(), list()
  macro, macrow = list(), list()

  macro_pl = [[] for _ in range(len(lcl))]
  macrow_pl = [[] for _ in range(len(lcl))]
  pooled_pl = [[] for _ in range(len(lcl))]

  """
  k-fold, the lcl approach is applied for each fold independently and the mean
  of the performance for each fold is the result
  """
  N = 5
  fold_idx = 1
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  for train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="Training cross-validation"): # set the same folds for each model

    # prediction for the current fold (each fold is independent)
    pred = np.zeros((len(test_index), len(labels_order))) # results for random forest multilabel classifier

    """ create y and x dataset for the current hierarchy """
    y = labels.copy() # labels for all functions in level (i.e., children)
    y_train, y_test = y.loc[train_index], y.loc[test_index] # train-test split of y
    # print(type(y), type(train_index))
    # y_train = y.loc[train_index] # train-test split of y

    X = data[fimp.col.tolist()].copy() # random forest dataset, the output of the previous level is different for both classifiers
    X_train, X_test = X.loc[train_index], X.loc[test_index] # train-test split for random forest

    """
    training and prediction for the current level. A special case arise for
    random forest when there is only ony child in the level, in that case the
    output of the rf is different
    """
    # random forest
    s_time = time()
    estimator.fit(X_train, y_train)
    _pred = estimator.predict_proba(X_test)
    for cidx, x, cls in zip(range(len(labels.columns)), _pred, estimator.classes_):
      pred[:,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
    f_time = time() - s_time

    """
    storing predictions and performance measures for the current fold.
    Predictions are saved in a different file for each fold to avoid loosing
    the fold indexes
    """
    pred_df = pd.DataFrame(pred, columns=labels_order)
    pred_df.index = test_index
    pred_df.to_csv("predictions/global/{0}_{1}.csv".format(root.replace(':',''), fold_idx))
    fold_idx += 1

    """
    compute the performance measures per level
    """
    u2level, idx_u2level = list(), list()
    for idxl, line in enumerate(lcl):
      _, children = line.strip().split('=') # level info parent_list= children_list
      children = children.strip().split(',')
      idx_children = [np.where(labels_order==x)[0][0] for x in children] # get the indices of children
      # all classes up to the current level
      u2level += children
      idx_u2level += idx_children

      measures = e.multiclass_classification_measures(pred[:,idx_u2level], labels[u2level].loc[test_index])

      macro_pl[idxl].append(measures[1])
      macrow_pl[idxl].append(measures[2])
      pooled_pl[idxl].append(measures[3])

    """
    compute the performance of the whole hierarchy
    """
    measures = e.multiclass_classification_measures(pred, y_test)
    macro.append(measures[1])
    macrow.append(measures[2])
    pooled.append(measures[3])
    etime.append(f_time)
    # break

  """
  final performance of the hierarchy is computed, mean of the performance for
  each fold. Then, the performance measures are saved
  """
  # print(pooled)
  pprint("Random Forest", np.mean(pooled), np.mean(macro), np.mean(macrow), np.mean(etime))

  results.loc[j] = [root, 1, np.mean(etime),
                    np.mean(pooled), np.mean(macro), np.mean(macrow)]
  results.to_csv("predictions/global.csv", index=False)

  for idxl in range(len(lcl)):
    results_pl.loc[(nlevels*j)+idxl] = [root, idxl+1,
                      np.mean(pooled_pl[idxl]), np.mean(macro_pl[idxl]), np.mean(macrow_pl[idxl])]
  results_pl.to_csv("predictions/global_pl.csv", index=False)
  nlevels = len(lcl)

  # break
