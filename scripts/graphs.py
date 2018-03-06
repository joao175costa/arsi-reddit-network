#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:00:55 2018

@author: togepi
"""

import pickle
import os
from collections import Counter

YEAR = 2007
MONTH = 10

os.chdir(str(YEAR))

with open("authors_"+str(YEAR)+'-'+str(MONTH)+'.pkl', "rb") as input_file:
    author_dict = pickle.load(input_file)

#%% NODES AND EDGES
nodes=[]
edges=[]
    
for sub in author_dict:
    print(sub)
    for link in author_dict[sub]:
        thread_authors = author_dict[sub][link]
        for author1 in thread_authors:
            if (author1 != '[deleted]'):
                nodes.append(author1)
                if (len(thread_authors)>1):
                    for author2 in thread_authors:
                        if ((author2 != '[deleted]') and (author2 != author1)):
                            edges.append((author1,author2))

nodes = dict(Counter(nodes))
edges = dict(Counter(edges))

#%%
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
nx.write_gexf(G,'gexf_'+str(YEAR)+'-'+str(MONTH))