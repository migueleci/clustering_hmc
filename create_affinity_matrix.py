import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%

# file = open('genes.txt','r')
# genes = [x.strip() for x in file.readlines()]
# ng = len(genes)
# file.close()
#
# data = pd.read_csv('gcn_edgelist.csv', names=['source','target','score'])
# data = data.astype({'source': int, 'target': int, 'score': np.float64})
# print('Data loaded ...')
# sr = data.shape[0]
#
# smax = data['score'].max()
# data['score'] = smax - data['score'] + 1
#
# data['score'].hist(bins=10)
# plt.savefig('{0}.pdf'.format('zscore'), format='pdf', dpi=600)
#
# data.to_csv('edgelist.csv', index=False)
#
# print('{0} edges'.format(sr))
# print('{0:.2f}% of the total number of edges are in the GCN'.format(((sr)*100)/(ng*ng)))

# %%

"""
Create new distance matrix using information from the gene co-expression
network (inverse of zscore, where 1 represents a strong relation and >1 is
weaker) and information from the associations between genes and functions.
The new distance between two genes will be the mean of the inverted zscore and
the proportion of shared functions.
"""

edgelist = pd.read_csv('data/edgelist.csv')
smax = edgelist['score'].max() - 1
total = len(edgelist)
edgelist = edgelist.to_dict('records')

# %%

gene_by_func = pd.read_csv("data/gene_term_list.csv")
def array2list(str):
  str = str.replace("\'",'\"').replace(" ", ", ")
  return json.loads(str)
gene_by_func['tList'] = gene_by_func.tList.apply(lambda t: array2list(t))
gene_by_func = dict([(x,y) for x,y in gene_by_func.itertuples(index=False)])
# gene_by_func

# %%

data = list()
for x in tqdm(edgelist, total=total):
  u,v,s = x['source'],x['target'],x['score']
  ns = (s - 1)/smax
  pf = 1
  if u in gene_by_func and v in gene_by_func:
    fu = gene_by_func[u]
    fv = gene_by_func[v]
    cfunc = np.intersect1d(fu, fv)
    afunc = np.union1d(fu, fv)
    pf = 1 - len(cfunc)/len(afunc)
  w = np.mean([ns, pf])
  data.append([u,v,w])

new_edgelist = pd.DataFrame(data, columns=['source','target','weight'])
new_edgelist.to_csv('data/affinity_edgelist.csv', index=False)

# %%
