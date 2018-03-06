#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:20:07 2018

@author: togepi
"""

#READ METADATA AND CREATE SOME INTERESTING GRAPHS
def read_metafile(filepath):
    with open(filepath) as f:
        content = f.readlines()
    data = {}
    for line in content:
        subname = line.split("\n")
        subname = subname[0].split(" ")
        data[subname[0]]=int(subname[1])
    return data

def create_metamatrix(paths):
    metadata = []
    for path in paths:
        metadata.append(read_metafile(path))
    return metadata

def get_comments_of_sub(metamatrix, sub):
    MONTHS = np.size(metamatrix)
    comments = np.zeros(MONTHS)
    for i in range(MONTHS):
        try:
            comments[i] = metamatrix[i][sub]
        except KeyError:
            comments[i]=0
    return comments

def get_all_comments(metamatrix):
    return get_comments_of_sub(metamatrix,'all')

def get_sub_number(metamatrix):
    MONTHS = np.size(metamatrix)
    subs = np.zeros(MONTHS)
    for i in range(MONTHS):
        subs[i]=len(metamatrix[i])
    return subs


# =============================================================================
# #%%
# import numpy as np
# import os
# 
# YEARS=['2007','2008']
# metapath = [] #stores path to all required metafiles
# 
# for year in YEARS:
#     path = year+'/meta/'    
#     metafiles = np.array(os.listdir(path))
#     metafiles = np.core.defchararray.add(path, metafiles)
#     metapath = np.concatenate((metapath,metafiles))
# 
# metadata = create_metamatrix(metapath)
# 
# #%%
# import matplotlib.pyplot as plt
# 
# total = get_all_comments(metadata)
# plt.plot(total)
# 
# total_subs = get_sub_number(metadata)
# plt.figure()
# plt.plot(total_subs)
# 
# sub = get_comments_of_sub(metadata,'programming')
# plt.figure()
# plt.plot(sub)
# =============================================================================
