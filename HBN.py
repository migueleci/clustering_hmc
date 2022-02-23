#!/usr/bin/env python
# coding: utf-8

# HBN method
# Miguel Romero, nov 3rd

import scipy.stats
import numpy as np
from tqdm import tqdm

# Finding the direct parent for a given term in a given heirarchy the direct parent are the
# group of terms that do not have any other descendant. Verified aug 25th, 2020
def direct_pa(go, terms, hie):
  go_id = np.nonzero(terms == go)[0]

  cand_pa_id = np.nonzero(hie[go_id,:])[1] # Find all candidate parent terms of term go
  sub_hie_idx = np.hstack((cand_pa_id, go_id))
  sub_hie = hie[np.ix_(sub_hie_idx, sub_hie_idx)]

  for i in range(len(cand_pa_id)):
    # If the number of descendants of candidate parent term i is greater than 1, it is NOT the
    # direct parent term. Discard all terms with more than 1 descendant
    if np.count_nonzero(sub_hie[:,i]) > 1:
      cand_pa_id[i] = -1 # Python index from 0, -1 means that is no the direct parent

  pa_id = cand_pa_id[cand_pa_id >= 0] # any value diff from -1 is related as a direct parent
  return terms[pa_id] # return id of parent(s) for term go


# Minimal Spanning Tree Algorithm on GO hierarchy. Verified may 24th
def mst(genes, terms, gene_by_go, go_by_go):
  hie = go_by_go[np.ix_(terms,terms)].copy()
  n = len(terms)

  # From the 2nd level (level directly below the root) and lower
  # discard terms with exactly one parent (thery are already a tree)
  leaf = terms[1:].copy()
  leaf[np.nonzero(hie.sum(axis=1) == 1)[0]-1] = -1
  leaf = leaf[leaf >= 0]

  if len(leaf) > 0: # If no 2nd level terms, keep the current hierarchy
    for i in range(len(leaf)):
      parents = direct_pa(leaf[i], terms, hie)
      if len(parents) > 1: # check if leaf has only one or multiple ancestors
        leaf_id = np.nonzero(terms == leaf[i])[0]
        p = np.zeros(len(parents))

        for j in range(len(parents)):
          a = gene_by_go[genes, leaf[i]]
          b = gene_by_go[genes, parents[j]]

          both, one = np.count_nonzero(a+b == 2), np.count_nonzero(a+b == 1)
          if both + one > 0:
            p[j] = both / (both + one)

        pa = parents[p == np.max(p)]
        if len(pa) > 1:
          pa = pa[np.random.randint(0, len(pa), 1)]
        pa_id = np.nonzero(terms == pa)[0]

        # Relabel the entries in hie
        hie[leaf_id,:] = 0
        ance_id1 = np.hstack((np.nonzero(hie[pa_id,:])[1], pa_id))
        hie[leaf_id, ance_id1] = 1
        ch_id = np.nonzero(hie[:, leaf_id])[0]
        if len(ch_id) > 0:
          hie[ch_id,:] = 0
          ance_id2 = np.hstack((np.nonzero(hie[leaf_id,:])[1], leaf_id))
          hie[np.ix_(ch_id, ance_id2)] = 1

  return hie

