import os
import sys
import csv
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from collections import Counter

# create path
def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass


path = 'data'
file = open('{0}/genes.txt'.format(path), 'r')
genes = [x.strip() for x in file.readlines()]
ng = len(genes)
file.close()

data = pd.read_csv('{0}/affinity_edgelist.csv'.format(path))
mat = np.zeros((ng, ng))

for u,v,s in tqdm(data.itertuples(index=False, name=None)):
	mat[u,v] = mat[v,u] = s
G_sparse = csr_matrix(mat)

for n_clusters in np.arange(10,101,10):
  sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed_nearest_neighbors', n_jobs=64, random_state=0).fit(G_sparse)
  labels = sc.labels_

  file = open('{0}/clustering/n{1}.csv'.format(path,n_clusters),'w')
  file.write('\n'.join([str(x) for x in labels]))
  file.close()


# print("Completeness: %0.3f" % metrics.completeness_score(df, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(df, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(df, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(df, labels))
# print("Silhouette Score: %0.3f" % metrics.silhouette_score(mat, list(labels), metric="precomputed"))

# file = open('{0}/labels.csv'.format(path),'w')
# file.write('\n'.join([str(x) for x in labels]))
# file.close()

# g = ig.Graph()
# g.add_vertices(ng)
# g.vs["name"] = genes
#
# edges = list(data[['source', 'target']].itertuples(index=False, name=None))
# g.add_edges(edges)
# g.es['weight'] = data['score']
#
# print(ig.summary(g))
#
# g.layout(layout='layout_fruchterman_reingold')
