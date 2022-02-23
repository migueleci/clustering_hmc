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

root_list = readListFile("ndata/roots.txt")
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
  path = "ndata/{0}".format(root.replace(':',''))
  labels = pd.read_csv("{0}/labels.csv".format(path)) # labels - true gene-function associations
  labels = labels.drop([root], axis=1) # drop info of root, not used
  labels_order = labels.columns
  gene_list = pd.read_csv("{0}/genes.csv".format(path), names=["idx"])
  gene_list = gene_list.idx.tolist()

  lcl = open("{0}/lcl.txt".format(path), 'r') # load lcl order
  lcl = [x.strip() for x in lcl.readlines()]
  parent_map = open("{0}/lcn.txt".format(path), 'r') # load parent of each function, used for cumulative probs.
  parent_map = dict([(l.split('=')[1].strip(),l.split('=')[0].strip()) for l in parent_map.readlines()])

  data = pd.DataFrame()
  for clust in next(os.walk('clustering'))[1]:
    clf = [x for x in os.listdir('clustering/{0}'.format(clust)) if 'GO:'+x.split('.')[0][2:] in labels_order]
    for file in clf:
      # if 'GO:'+file.split('.')[0][2:] in leaves: ###############################
      tmp = pd.read_csv('clustering/{0}/{1}'.format(clust,file))
      data['{0}_{1}'.format(clust,file.split('.')[0][2:])] = tmp.label
  data = data.loc[gene_list].reset_index(drop=True)
  # feature_importance = pd.read_csv('{0}/features.csv'.format(path))

  # load both models and evaluation class
  estimator = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=seed) # Random forest classifier
  e = Evaluate()


  """
  k-fold, the lcl approach is applied for each fold independently and the mean
  of the performance for each fold is the result
  """
  N = 10
  feature_importance_ = [None for _ in lcl]
  columns_ = [None for _ in lcl]
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  for train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="SHAP"): # set the same folds for each model

    """
    for each level in the hierarchy a model is trained for all functions in it
    considering the information from the ancestors in the previous level
    """
    count = 0
    for line in lcl:
      ancestors, children = line.strip().split('=') # level info parent_list= children_list
      ancestors = ancestors.strip().split(',')
      children = children.strip().split(',')
      ancestors, children = np.array(ancestors), np.array(children)
      curr_parent = [parent_map[x] for x in children if parent_map[x] != root]

      idx_children = [np.where(labels_order==x)[0][0] for x in children] # get the indices of children
      if root not in ancestors:
        idx_ancestors = [np.where(labels_order==x)[0][0] for x in ancestors] # get the indices of parents

      """
      create y dataset for the current level, the dataset changes according to the
      functions to be predicted, the parents of functions in the level, and the
      nodes that are related to any of the parents
      """
      y = labels[children] # labels for all functions in level (i.e., children)
      _train_index = train_index.copy()
      if root not in ancestors:
        y_parents = labels[ancestors].loc[train_index]
        _train_index = y_parents.loc[(y_parents != 0).any(axis=1)].index.to_numpy()
      y_train = y.loc[_train_index] # train-test split of y

      """
      create X dataset for the level using the initial datasets of the
      hierarchy, predictions from parents of the previous level are used as
      features for the current
      """
      cols = [c for c in data.columns if "GO:"+c[4:] in children or "GO:"+c[4:] in ancestors]
      X = data[cols].copy() # random forest dataset, the output of the previous level is different for both classifiers
      X_train, X_test = X.loc[_train_index], X.loc[test_index] # train-test split for random forest

      """
      training and prediction for the current level. A special case arise for
      random forest when there is only ony child in the level, in that case the
      output of the rf is different
      """
      # random forest
      s = time()
      estimator.fit(X_train, y_train if len(children) > 1 else y_train[children[0]].to_list())
      explainer = shap.TreeExplainer(estimator)
      shap_values = np.array(explainer.shap_values(X_test))

      fimp = np.zeros(X.shape[1])
      for fidx in range(y.shape[1]):
        vals = abs(shap_values[fidx*2+1]).mean(0)
        fimp += vals

      fimp = fimp / y.shape[1]
      if feature_importance_[count] is None:
        feature_importance_[count] = np.zeros(X.shape[1])
      feature_importance_[count] += fimp
      columns_[count] = cols

      count += 1

  for level in range(len(lcl)):
    fimp = pd.DataFrame(list(zip(columns_[level], feature_importance_[level])), columns=['col','val'])
    fimp.val = fimp.val / N
    fimp.sort_values(by=['val'],ascending=False,inplace=True)
    fimp = fimp[fimp.val > 0]
    fimp.to_csv('{0}/features_{1}.csv'.format(path,level), index=False)

    thr = fimp.val.sum() * 0.9
    fimp['cum']= fimp.val.cumsum()
    fimp = fimp[fimp.cum <= thr]
    feature_importance_[level] = fimp.col.tolist()




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
    pred = np.zeros(labels.shape) # results for random forest multilabel classifier
    rpred = np.zeros(labels.shape) # raw results for random forest multilabel classifier

    """
    for each level in the hierarchy a model is trained for all functions in it
    considering the information from the ancestors in the previous level
    """
    count = 0
    f_time = 0
    for line in lcl:
      ancestors, children = line.strip().split('=') # level info parent_list= children_list
      ancestors = ancestors.strip().split(',')
      children = children.strip().split(',')
      ancestors, children = np.array(ancestors), np.array(children)
      curr_parent = [parent_map[x] for x in children if parent_map[x] != root]
      # print(ancestors,"-->",children)

      idx_children = [np.where(labels_order==x)[0][0] for x in children] # get the indices of children
      if root not in ancestors:
        idx_ancestors = [np.where(labels_order==x)[0][0] for x in ancestors] # get the indices of parents

      """
      create y dataset for the current level, the dataset changes according to the
      functions to be predicted, the parents of functions in the level, and the
      nodes that are related to any of the parents
      """
      y = labels[children] # labels for all functions in level (i.e., children)
      _train_index = train_index.copy()
      if root not in ancestors:
        y_parents = labels[ancestors].loc[train_index]
        _train_index = y_parents.loc[(y_parents != 0).any(axis=1)].index.to_numpy()
      y_train = y.loc[_train_index] # train-test split of y

      """
      create X dataset for the level using the initial datasets of the
      hierarchy, predictions from parents of the previous level are used as
      features for the current
      """
      # cols = [c for c in feature_importance.col if "GO:"+c[4:] in children]
      # fimp = feature_importance[feature_importance.col.isin(cols)].reset_index(drop=True)
      # thr = fimp.val.sum() * 0.9
      # fimp['cum'] = fimp.val.cumsum()
      # fimp = fimp[fimp.cum <= thr]

      X = data[feature_importance_[count]].copy() # random forest dataset, the output of the previous level is different for both classifiers
      # X = data[fimp.col.tolist()].copy() # random forest dataset, the output of the previous level is different for both classifiers
      # if it is not the first level the prediction of previous level is loaded into datasets
      for i in range(len(ancestors)):
        if ancestors[i] in curr_parent:
          X[ancestors[i]] = pred[:,idx_ancestors[i]]
          X[ancestors[i]+"_raw"] = rpred[:,idx_ancestors[i]]

      X_train, X_test = X.loc[_train_index], X.loc[test_index] # train-test split for random forest

      """
      training and prediction for the current level. A special case arise for
      random forest when there is only ony child in the level, in that case the
      output of the rf is different
      """
      # random forest
      s = time()
      estimator.fit(X_train, y_train if len(children) > 1 else y_train[children[0]].to_list())
      _pred = estimator.predict_proba(X_test)
      if len(children) > 1:
        for cidx, x, cls in zip(idx_children, _pred, estimator.classes_):
          pred[test_index,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
          rpred[test_index,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
      else:
        _pred = _pred[:,0] if estimator.classes_[0] == 1 else 1- _pred[:,0]
        pred[np.ix_(test_index,idx_children)] = np.reshape(_pred, (len(test_index), 1))
        rpred[np.ix_(test_index,idx_children)] = np.reshape(_pred, (len(test_index), 1))
      f_time += time() - s

      """
      predictions from the current level is used for prediction in the next
      level, the model the current level (which has noot been modified) is used
      to predict the training set, so the test set is unseen data
      """
      _pred = estimator.predict_proba(X_train)
      if len(children) > 1:
        for cidx, x, cls in zip(idx_children, _pred, estimator.classes_):
          pred[_train_index,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
          rpred[_train_index,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
      else:
        _pred = _pred[:,0] if estimator.classes_[0] == 1 else 1 - _pred[:,0]
        pred[np.ix_(_train_index,idx_children)] = np.reshape(_pred, (len(_train_index), 1))
        rpred[np.ix_(_train_index,idx_children)] = np.reshape(_pred, (len(_train_index), 1))

      """
      cumulative probabilities are computed for the functions in the current
      level, so the true-path rule is guaranteed (hierarchy consistency)
      """
      if root not in ancestors: # compute cumulative probabilities for each function in level
        for i, c in enumerate(children):
          cid = idx_children[i]
          pid = idx_ancestors[np.where(ancestors==parent_map[c])[0][0]] # get index of its parent
          pred[:,cid] = pred[:,cid] * pred[:,pid] # compute cumprob for random forest
          # pred[:,cid] = np.minimum(pred[:,cid], pred[:,pid])

      count += 1

    """
    storing predictions and performance measures for the current fold.
    Predictions are saved in a different file for each fold to avoid loosing
    the fold indexes
    """
    pred_df = pd.DataFrame(rpred[test_index,:], columns=labels_order)
    pred_df.index = test_index
    # pred_df.to_csv("npred/lcl/{0}_{1}.csv".format(root.replace(':',''), fold_idx))
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

      measures = e.multiclass_classification_measures(pred[np.ix_(test_index,idx_u2level)], labels[u2level].loc[test_index])
      macro_pl[idxl].append(measures[1])
      macrow_pl[idxl].append(measures[2])
      pooled_pl[idxl].append(measures[3])

    """
    compute the performance of the whole hierarchy
    """
    measures = e.multiclass_classification_measures(pred[test_index,:], labels.loc[test_index])
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
  print("Models trained: {0}".format(count))

  results.loc[j] = [root, count, np.mean(etime),
                    np.mean(pooled), np.mean(macro), np.mean(macrow)]
  results.to_csv("npred/lcl.csv", index=False)

  for idxl in range(len(lcl)):
    results_pl.loc[(nlevels*j)+idxl] = [root, idxl+1,
                      np.mean(pooled_pl[idxl]), np.mean(macro_pl[idxl]), np.mean(macrow_pl[idxl])]
  results_pl.to_csv("npred/lcl_pl.csv", index=False)
  nlevels = len(lcl)

  break
