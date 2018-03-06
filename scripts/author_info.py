#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:02:34 2018

@author: togepi
"""

def get_subs(YEAR, MONTH):
    path = str(YEAR)+'/meta/meta_RC_'+str(YEAR)+'-'+str(MONTH)
    meta = read_metafile(path)
    del meta['all']
    
    path = str(YEAR)+'/subs/subs_RC_'+str(YEAR)+'-'+str(MONTH)
    os.chdir(path)
    
    import operator
    sorted_meta = sorted(meta.items(), key=operator.itemgetter(1))
    return [sub[0] for sub in sorted_meta]

def get_unique_authors(sub_dump):
    authors = []
    with open(sub_dump) as f:
        for line in f:
            authors.append(json.loads(line)['author'])  
    unique_authors = Counter(authors)
    del unique_authors['[deleted]']
    return dict(unique_authors)
    
def get_authors(YEAR, MONTH):
    subs_dumps = get_subs(YEAR, MONTH)
    author_matrix = []
    for sub in subs_dumps:
        authors = get_unique_authors(sub)
        author_matrix.append(authors)
    return author_matrix, subs_dumps

def get_common_author_matrix(YEAR, MONTH):
    author_dict, subs = get_authors(YEAR, MONTH)
    nSubs = len(author_dict)
    print ('Total number of subs: ', nSubs)
    CMA = np.zeros((nSubs, nSubs))
    for sub_i in range(nSubs):
        print(sub_i,'/',nSubs)
        for sub_j in range(nSubs):
            if (sub_i == sub_j):
                continue
            elif (sub_i < sub_j):
                sub_i_authors = author_dict[sub_i].keys()
                sub_j_authors = author_dict[sub_j].keys()
                auth_counter = 0
                for i_author in sub_i_authors:
                    for j_author in sub_j_authors:
                        if (i_author==j_author):
                            auth_counter+=1
                try:
                    overlap=auth_counter
                    CMA[sub_i,sub_j] = overlap
                    CMA[sub_j,sub_i] = overlap
                except ZeroDivisionError:
                    pass
    return CMA, subs
                
    
#%%
import json
import os
import numpy as np
import pickle
from read_metadata import read_metafile
from collections import Counter

year = [2007,2008,2009]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']

for y in year:
    for m in month:
        print(y,m)
        try:
            shared_authors, subs = get_common_author_matrix(y,m)
            os.chdir('..')
            os.chdir('..')
            with open('shared_authors_'+str(y)+'-'+str(m) + '.pkl', 'wb') as f:
                pickle.dump(shared_authors,f,pickle.HIGHEST_PROTOCOL)
                
            with open('subs_'+str(y)+'-'+str(m)+'.pkl','wb') as f:
                pickle.dump(subs, f, pickle.HIGHEST_PROTOCOL)
                
            os.chdir('..')    
        except FileNotFoundError as e:
            print(e.strerror)
        
#%%
import matplotlib.pyplot as plt

zero_mask = (shared_authors == 0)
masked = np.ma.masked_where(zero_mask, shared_authors)
plt.imshow(masked, cmap='plasma')
    
