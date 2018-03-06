#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:30:54 2018

@author: togepi
"""
import json
import os

def dump_reddit(REDDIT_DUMP):

    year = REDDIT_DUMP[3:7]
    os.chdir(year)
    
    data = []
    with open(REDDIT_DUMP) as f:
        for line in f:
            data.append(json.loads(line))
    print('Loaded file')
            
    #%%
    import numpy as np
    
    size = np.size(data)
    subs = []
    for i in range(size):
        subs.append(data[i]['subreddit'])
    print('nr of comments: ', size)
    
    #%%
    from collections import Counter
    
    distinct_subs = list(Counter(subs))
    nr_subs = np.size(distinct_subs)
    comments_subs = [[] for i in range(nr_subs)]
    for i in range(size):
        for j in range(nr_subs):
            comment = data[i]['subreddit']
            if (distinct_subs[j] == comment):
                comments_subs[j].append(data[i])
    print('distinct subs: ', nr_subs)
    
    #%% METADATA FILE
    metafile = 'meta_'+REDDIT_DUMP
    with open(metafile,'w') as meta:
        meta.write('all ' + str(size) + '\n')
        for i in range(nr_subs):
            sub_name = distinct_subs[i]
            sub_com_total = np.size(comments_subs[i])
            meta.write(sub_name + ' '+ str(sub_com_total) + '\n')
    print('Metadata file created')
    
    #%%
    new_folder ='subs_' + REDDIT_DUMP
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    os.chdir(new_folder)
    print('Folder created')
    
    for i in range(nr_subs):
        filename = distinct_subs[i]
        with open(filename, 'w') as outfile:
            for line in range(np.size(comments_subs[i])):      
                single_comment = json.dumps(comments_subs[i][line])
                outfile.write(single_comment+'\n')
    print('Files created')
    os.chdir('..')
    os.chdir('..')