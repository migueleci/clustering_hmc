# %%

import os
import json
import numpy as np
import igraph as ig
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.semantic import semantic_similarity

import scipy.stats as stats                # for fisher test
import statsmodels.stats.multitest as smt  # for pvalue adjustment

pd.set_option("display.precision", 2)

# %%

# df = pd.read_csv('data/functions_gene.csv', names=["id",'functions'])
godag = GODag("data/go-basic.obo")
ndf = pd.read_csv("data/gene_term_list.csv")
def array2list(str):
  str = str.replace("\'",'\"').replace(" ", ", ")
  return json.loads(str)
ndf.tList = ndf.tList.apply(lambda t: array2list(t))

ufunc = list()
for x in ndf.tList.tolist(): ufunc += x
ufunc = list(set(ufunc))

gene4func = dict([(x,0) for x in ufunc])
for x in ndf.tList.tolist():
  for f in x:
    gene4func[f] += 1

# %%

def GO_enrichment():
  '''
  :param com2genes: dictionary where keys are community ids and values are list of genes belonging to the corresponding community
  :param GO2genes: dictionary where keys are GO terms and values are list of genes with the corresponding GO annotation
  :param gene2GOs: dictionary where keys are genes and values are list of GO terms annotated to the corresponding gene
  :param n: total number of genes in the background set
  '''

  nldf = ldf[ldf.label >= 0]
  nbr_coms = len(nldf.label.unique())

  ans = []
  for m in range(nbr_coms):
    module = nldf[nldf.label == m]
    nbr_genes_in_module = len(module)

    # GO terms present in the module
    module_GOs = list()
    mndf = ndf[ndf.ID.isin(module.index)]
    for x in mndf.tList.tolist(): module_GOs += x
    module_GOs = set(module_GOs)

    for go in module_GOs:
      nbr_genes_in_GO = gene4func[go]

      a = np.sum([1 if go in x else 0 for x in mndf.tList])
      b = nbr_genes_in_GO - a
      c = nbr_genes_in_module - a
      d = len(ldf) - nbr_genes_in_GO - c

      _, pvalue = stats.fisher_exact([[a,b],[c,d]])

      # mod, go, pval
      ans.append([m,go,pvalue])

  goedf = pd.DataFrame(ans, columns=['module','GO_id','pvalue'])
  return goedf

def FDR(tmpdf):
  '''
  :param df: DataFrame from GO_enrichment function
  '''
  fdr = []
  coms = tmpdf.module.unique()
  for m in coms:
    try:
      pval_adj = smt.multipletests(pvals = tmpdf[tmpdf.module==m].pvalue, method = 'fdr_bh')[1]
      fdr += list(pval_adj)
    except:
      fdr += tmpdf[tmpdf.module==m].pvalue.tolist()

  tmpdf['fdr'] = fdr
  return tmpdf

def enriched_modules():
  goedf = GO_enrichment()
  goedf2 = FDR(goedf)
  goedf2 = goedf2[goedf2.fdr < 0.05]
  return goedf2
  # return goedf2.module.nunique()

# %%

'''
Create files to compare with xgb results

'''
def create_files(labels, df, folder):
  total = 0
  for func in df.GO_id.unique():
    arr = np.zeros(len(labels))
    mods = df[df.GO_id==func].module.tolist()
    idx = labels[labels.label.isin(mods)].index
    arr[idx] = 1
    res = pd.DataFrame()
    res['label'] = arr
    res.to_csv('{0}/{1}.csv'.format(folder, func.replace(':','')), index=False)
    print('{0} --> {1}'.format(func, np.sum(arr)))
    total += np.sum(arr)
  return total

# create file of pvalues
def create_files_pvalues(labels, df, folder):
  total = 0
  for func in df.GO_id.unique():
    arr = np.zeros(len(labels))
    for mod in df[df.GO_id==func].module:
      pvalue = df[(df.GO_id==func) & (df.module==mod)].pvalue
      idx = labels[labels.label==mod].index
      arr[idx] = 1 - pvalue
    # print('{0} --> {1}'.format(func, np.count_nonzero(arr)))
    res = pd.DataFrame()
    res['label'] = arr
    res.to_csv('{0}/{1}.csv'.format(folder, func.replace(':','')), index=False)
    total += np.sum(arr)
  return total

def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass

# %%

n_clusters = [_ for _ in range(10,101,10)]
n_clusters += ['n{0}'.format(x) for x in n_clusters]
# n_clusters = n_clusters[:2]
files = ['{0}.csv'.format(x) for x in n_clusters]

for nc, file in tqdm(zip(n_clusters, files), total=len(n_clusters)):
  ldf = pd.read_csv('clustering/{0}'.format(file), names=["label"])
  n = len(ldf.label.unique())
  goedf2 = enriched_modules()
  # print('\n#### {0}\n'.format(nc))
  # print(goedf2.module.value_counts(sort=True))
  # print(ldf.label.value_counts(sort=True))

  outpath = 'clustering/{0}'.format(nc)
  create_path(outpath)
  total = create_files_pvalues(ldf, goedf2, outpath)
