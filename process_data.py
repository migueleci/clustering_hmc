import os
import json
import multiprocessing

import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt

from HBN import *

from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot

# %%

godag = GODag("data/go-basic.obo")
file = open("data/genes.txt", 'r')
genes = [int(x.strip()) for x in file.readlines()]
gIDs = dict([(x,i) for i,x in enumerate(genes)])
file.close()

# %%
# DAVID db gene-term associations

g2t_new = pd.read_csv("data/ENTREZ_GENE_ID2GOTERM_BP_ALL.txt","\t",header=None,names=["Entrez","GOb"])
g2t_new = g2t_new[g2t_new.Entrez.isin(genes)].reset_index(drop=True)
g2t_new["ID"] = g2t_new.Entrez.apply(lambda x: gIDs[x])
g2t_new["GO"] = g2t_new.GOb.apply(lambda x: x.split('~')[0])

g2t_new.drop("GOb", axis=1, inplace=True)
g2t_new.drop("Entrez", axis=1, inplace=True)
print("GO terms: {0}".format(len(g2t_new.GO.unique())))
print("Gene-GO: {0}".format(g2t_new.shape[0]))

# %%
# Old gene-term associations - source unknown

def array2list(str):
  str = str.replace("\'",'\"')
  return json.loads(str)

g2t_old = pd.read_csv("data/functions_old.csv", header=None, names=["gID", "tList"])
g2t_old_it = g2t_old.to_dict('records')

data = list()
for line in g2t_old_it:
  for term in line["tList"].split():
    data.append((line["gID"], term))
g2t_old = pd.DataFrame(data, columns=["ID","GO"])
g2t_old = g2t_old.drop_duplicates().reset_index(drop=True)
print("GO terms: {0}".format(len(g2t_old.GO.unique())))
print("Old Gene-GO: {0}".format(g2t_old.shape[0]))

# %%
# Concatenate both dataframes and keep only biological processes

g2t = pd.concat([g2t_new, g2t_old])
g2t = g2t.drop_duplicates().reset_index(drop=True)
g2t = g2t[g2t.GO.isin(godag) & (g2t.GO != "GO:0008150")].reset_index(drop=True)
# g2t.to_csv("data/gene_term.csv", index=False)

# %%

g2t = pd.read_csv("data/gene_term.csv")
terms = g2t.GO.unique()
print("Number of terms (non-obsolet): {0}".format(len(terms)))
print("Number of relations: {0}".format(len(g2t)))

# %%
# Get terms ancestral relations for hierarchy creation

isa = list()
for t in tqdm(terms):
  q = deque()
  for p in godag[t].parents:
    q.append((t, p.id))

  while len(q) > 0:
    c, p = q.pop()
    if p != "GO:0008150":
      isa.append((c,p))
      for gp in godag[p].parents:
        q.append((p, gp.id))

isa = pd.DataFrame(isa, columns=['Child','Parent'])
isa = isa.drop_duplicates().reset_index(drop=True)
# isa.to_csv("data/isa.csv", index=False)

all_terms = np.union1d(np.union1d(isa.Child, isa.Parent), terms)
term_def = pd.DataFrame()
term_def["Term"] = all_terms
term_def["Desc"] = [godag[t].name for t in all_terms]
# term_def.to_csv("data/term_def.csv", index=False)

print('Number of terms: {0}'.format(len(all_terms)))
print('Number of relations: {0}'.format(len(isa)))

# %%

data_gcn = pd.read_csv("data/edgelist.csv")
ng = len(genes)
terms = all_terms.copy()
nt, tIDs = len(terms), dict([(t,i) for i,t in enumerate(terms)])

print('**Initial data**')
print('Genes: \t\t{0:8d}'.format(ng))
print('Gene annot.: \t{0:8d}'.format(len(g2t)))
print('Co-expression: \t{0:8.0f}'.format(len(data_gcn)))
print('GO terms: \t{0:8d}'.format(nt))
print('GO hier.: \t{0:8d}'.format(len(isa)))

# %%
# GCN matrix
# ng:number of genes, gIDs:gene index map
# gcn = np.zeros((ng,ng))
# for edge in tqdm(data_gcn):
#   u, v = gIDs[edge["source"]], gIDs[edge["target"]]
#   gcn[u][v] = gcn[v][u] = 1

# go by go matrix
# nt:number of terms, tIDs:term index map
go_by_go = np.zeros((nt,nt))
for edge in tqdm([tuple(x) for x in isa.to_numpy()]):
  u, v = tIDs[edge[0]], tIDs[edge[1]]
  go_by_go[u,v] = 1

# compute the transitive closure of the ancestor of a term (idx)
def ancestors(term):
  tmp = np.nonzero(go_by_go[term,:])[0]
  ancs = list()
  while len(tmp) > 0:
    tmp1 = list()
    for i in tmp:
      ancs.append(i)
      tmp1 += np.nonzero(go_by_go[i,:])[0].tolist()
    tmp = list(set(tmp1))
  return ancs

# gene by go matrix
gene_by_go = np.zeros((ng,nt))
for edge in tqdm([tuple(x) for x in g2t.to_numpy()]):
  u, v = edge[0], tIDs[edge[1]]
  gene_by_go[u,v] = 1
  gene_by_go[u,ancestors(v)] = 1

print()
print('**Final data**')
print('Genes: \t\t{0:8}'.format(ng))
print('Gene annot.: \t{0:8}'.format(np.count_nonzero(gene_by_go)))
print('Co-expression: \t{0:8.0f}'.format(len(data_gcn)))
print('GO terms: \t{0:8}'.format(nt))
print('GO hier.: \t{0:8.0f}'.format(np.sum(go_by_go)))

# %%
# Create file with list of functions per gene

data = list()
for i in tqdm(range(ng)):
  tlist = np.nonzero(gene_by_go[i,:])[0]
  if len(tlist) > 0:
    data.append((i,terms[tlist]))
termList = pd.DataFrame(data, columns=["ID","tList"])
# termList.to_csv("data/gene_term_list.csv", index=False)

# %%
#####################################
# 2. Prepare term data for prediction
#####################################

# Graph for subhiearchies creation
g2g_edg = np.transpose(np.nonzero(np.transpose(go_by_go))).tolist()
g2g = nx.DiGraph()
g2g.add_nodes_from(np.arange(nt))
g2g.add_edges_from(g2g_edg)
print('GO graph (all terms): nodes {0}, edges {1}'.format(g2g.number_of_nodes(), g2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(g2g)))

# Prune terms according to paper, very specific and extremes with little to
# no information terms are avoided. Select genes used for prediction
# Accoding to restriction 5 <= genes annotated <= 300
ft_idx = list() # list of terms filtered according to the previous explanation
for i in range(nt):
  if 200 <= np.count_nonzero(gene_by_go[:,i]):
    ft_idx.append(i)
print('Number of filtered terms: {0}'.format(len(ft_idx)))

# Including the ancestor of the selected terms
pt_idx = list(ft_idx)
for i in ft_idx:
  pt_idx += np.nonzero(go_by_go[i,:])[0].tolist()
pt_idx = np.array(sorted(list(set(pt_idx))))
print('Number of filtered terms incl. parents: {0}'.format(len(pt_idx)))

# Subgraph from terms to predict
sub_go_by_go = go_by_go[np.ix_(pt_idx,pt_idx)].copy()
sg2g_edg = np.transpose(np.nonzero(np.transpose(sub_go_by_go))).tolist()
sg2g = nx.DiGraph()
sg2g.add_nodes_from(np.arange(len(pt_idx)))
sg2g.add_edges_from(sg2g_edg)
print('GO subgraph (pred terms): nodes {0}, edges {1}'.format(sg2g.number_of_nodes(), sg2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(sg2g)))

# x = gene_by_go.sum(0)
# x = [a for a in x if a <= 1000 ]
# plt.figure(figsize=(20,10))
# plt.hist(x, bins=100)
# plt.show()

# find possible root terms in go subgraph
proot_idx = list() # possible hierarchy roots
for i in range(len(pt_idx)):
  if np.count_nonzero(sub_go_by_go[i,:]) == 0: # terms wo ancestors
    proot_idx.append(i)
proot_idx = np.array(proot_idx)
print('Number of roots in GO subgraph: {0}'.format(len(proot_idx)))

# convert a bfs object to a list
def nodes_in_bfs(bfs, root):
  nodes = sorted(list(set([u for u,v in bfs] + [v for u,v in bfs])))
  nodes = np.setdiff1d(nodes, [root]).tolist()
  nodes = [root] + nodes
  return nodes

# detect isolated terms and create sub-hierarchies
hpt = list() # terms to predict and all terms in hierarchy
hroot_idx = list()
for root in proot_idx:
  bfs = nx.bfs_tree(sg2g, root).edges()

  if len(bfs) > 0: # if no isolated term
    hroot_idx.append(pt_idx[root])
    hpt.append(pt_idx[nodes_in_bfs(bfs, root)])

hroot_idx = np.array(hroot_idx)
len_hpt = [len(x) for x in hpt]
print('Number of isolated terms: {0}'.format(len(proot_idx)-len(hroot_idx)))
print('Number of sub-hierarchies: {0}'.format(len(hroot_idx)))
print('Average terms in sub-hierarchies: {0:.2f} [{1}-{2}]'.format(
  np.mean(len_hpt),
  np.min(len_hpt),
  np.max(len_hpt)))

# %%

# list sub-hierarchies
df_subh = pd.DataFrame(columns=['Root_idx','Root','Terms','Genes','Desc','Level'])
for i, rid in enumerate(hroot_idx):
  root = terms[rid]
  data = [rid, root, len(hpt[i])] # number of terms to predict in sub-hier.
  data += [np.count_nonzero(gene_by_go[:,rid])] # number of genes in sub.
  data += [term_def[term_def.Term==root].Desc.tolist()[0], godag[root].level]
  df_subh.loc[i] = data

df_subh = df_subh.sort_values(by=['Terms','Genes'], ascending=False).reset_index(drop=True)
# df_subh.to_csv('data/hierarchies.csv', index=False)
df_subh

# %%

# sub-hierarchies used for prediction
test_df_subh = df_subh[df_subh.Terms >= 9].sort_values(by=['Terms','Genes'], ascending=True).reset_index(drop=True)
test_r = test_df_subh.Root.tolist()
test_rid = test_df_subh.Root_idx.tolist()

test_hpt = list()
for i, root in enumerate(test_rid):
  idx = np.where(hroot_idx==root)[0][0]
  test_hpt.append(hpt[idx])

# test_df_subh.to_csv("data/fhierarchies.csv", index=False)
test_df_subh

# %%

def list2file(l, name):
  file = open(name, 'w')
  file.write('\n'.join([str(x) for x in l]))
  file.close()

def create_path(path):
  try: os.makedirs(path)
  except: pass

list2file(test_r, "data/roots.txt")
for x, root in zip(test_hpt, test_r):
  path = "data/{0}".format(root.replace(':',''))
  create_path(path)
  list2file(terms[x], "{0}/terms.txt".format(path))

# %%

train_times = pd.DataFrame(columns=["Root","Terms","Genes","LCN","LCPN","LCL", "Global"])

for j, (x, root) in tqdm(enumerate(zip(test_hpt, test_rid))):
  hgenes = np.nonzero(gene_by_go[:,root])[0]
  hterms = terms[x] # terms to predict in hierarchy

  # Conver DAG to tree, will be used for prediction
  tree = mst(hgenes, x, gene_by_go.copy(), go_by_go.copy())
  hg2g = np.zeros((len(hterms),len(hterms)))
  for i, idx in enumerate(x):
    parents = direct_pa(idx, x, tree)
    parents = [np.where(x == p)[0][0] for p in parents]
    hg2g[i, parents] = 1

  q = deque()
  q.append((0,0)) # parent, level
  parents, level = 0, 0

  lcn = list()
  lcpn = list()
  lcl, lastl, pterms, cterms = list(), 0, list(), list()

  while len(q) > 0:
    pos, l = q.popleft()
    children = np.nonzero(hg2g[:,pos])[0]

    # lcl order of prediction
    if lastl != l:
      lastl = l
      lcl.append("{0}= {1}".format(','.join(pterms), ','.join(cterms)))
      pterms, cterms = list(), list()
    pterms.append(hterms[pos])
    cterms += list(hterms[children])

    if len(children) > 0: # is a parent
      lcpn.append(("{0}= {1}".format(hterms[pos], ','.join(hterms[children])))) # save lcpn order of prediction

      parents += 1
      for c in children:
        lcn.append(("{0}= {1}".format(hterms[pos], hterms[c]))) # save lcn order of prediction
        q.append((c,l+1))

    level = max(level, l)

  train_times.loc[j] = [hterms[0], len(x), len(hgenes), len(x)-1, parents, level, 1]

  path = "data/{0}".format(hterms[0].replace(':',''))
  create_path(path)
  list2file(lcn, "{0}/lcn.txt".format(path))
  list2file(lcpn, "{0}/lcpn.txt".format(path))
  list2file(lcl, "{0}/lcl.txt".format(path))

train_times

# %%
from matplotlib import rc
rc('font', family='serif', size=18)
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(7,6))
train_times[["Root","LCN","LCPN","LCL","Global"]].plot(ax=ax, style='--o')
plt.xticks(np.arange(len(train_times)), train_times.Root.tolist(), rotation=90)
plt.grid(visible=True, axis='both', ls='--', alpha=0.5)
plt.ylim(0,100)
plt.xlabel('Sub-hierarchy')
plt.ylabel('Number of classifiers')
# plt.title('Number of iterations per HMC meth')
plt.legend(labels=[r'\textit{lcn}', r'\textit{lcpn}', r'\textit{lcl}', 'global'])
plt.tight_layout()
plt.savefig('data/training_times.pdf', format='pdf', dpi=600)
plt.close()

# %%

fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].plot(train_times.Root, train_times.Terms, '--o')
ax[1].plot(train_times.Root, train_times.Genes, '--o')
ax[0].set_title('Number of GO terms per hierarchy')
ax[1].set_title('Number of genes per hierarchy')
ax[0].set_xticks(np.arange(len(train_times)))
ax[1].set_xticks(np.arange(len(train_times)))
ax[0].set_xticklabels(train_times.Root.tolist(), rotation=90)
ax[1].set_xticklabels(train_times.Root.tolist(), rotation=90)
ax[0].grid(visible=True, axis='both', ls='--', alpha=0.5)
ax[1].grid(visible=True, axis='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('data/hierarcy_desc.pdf', format='pdf', dpi=300)
plt.close()

# %%

for i, (root, hterms) in enumerate(zip(test_rid, test_hpt)):

  term = terms[root]
  hgenes = np.nonzero(gene_by_go[:,root])[0]

  # create sub matrix terms_hier_idx hierarchy
  sm_gene_by_go = gene_by_go[np.ix_(hgenes,hterms)].copy()

  sh_df = pd.DataFrame()
  for i, trm in enumerate(terms[hterms]):
    sh_df[trm] = pd.Series(sm_gene_by_go[:,i])
  path = "data/{0}".format(term.replace(':',''))
  list2file(hgenes, '{0}/genes.csv'.format(path))
  sh_df.to_csv('{0}/labels.csv'.format(path), index=False)

  print('{0}\tTerms: {1:3}\tGenes: {2:4}'.format(term, len(hterms), len(hgenes)))
