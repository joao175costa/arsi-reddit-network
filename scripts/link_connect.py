#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:24:12 2018

@author: togepi
"""

import json
import os
import operator
import pickle
from read_metadata import read_metafile
from collections import Counter

    
def get_subs(YEAR, MONTH):
    path = str(YEAR)+'/meta/meta_RC_'+str(YEAR)+'-'+str(MONTH)
    meta = read_metafile(path)
    del meta['all']
    
    subs_path = str(YEAR)+'/subs/subs_RC_'+str(YEAR)+'-'+str(MONTH)
    os.chdir(subs_path)
    
    sorted_meta = sorted(meta.items(), key=operator.itemgetter(1))
    return sorted_meta

def get_links_of_sub(sub):
    links = []
    with open(sub) as f:
        for line in f:
            links.append(json.loads(line)['link_id'])
    unique_links = list(Counter(links))        
    return unique_links

def get_links(YEAR, MONTH):
    links_dict = {}
    subs = get_subs(YEAR, MONTH)
    for sub in subs:
        links_dict[sub[0]] = get_links_of_sub(sub[0])
    return links_dict

def get_authors_in_link(sub, link_id):
    authors_in_link = {}
    with open(sub) as f:
        for line in f:
            comment = json.loads(line)
            if (comment['link_id'] == link_id):
                authors_in_link[comment['author']] = authors_in_link.get(comment['author'],0) + 1
    return authors_in_link

def get_authors_of_sub(sub):
    authors_in_sub = {}
    links_list = get_links_of_sub(sub)
    for link_id in links_list:
        authors_in_sub[link_id] = get_authors_in_link(sub,link_id)
    return authors_in_sub
    
def get_authors_all(YEAR, MONTH):
    total = {}
    subs = get_subs(YEAR, MONTH)
    i=0
    for sub in subs:
        authors_subs = {}
        print(sub,i,'/',len(subs))
        subfile = sub[0]
        with open(subfile) as sub_dump:
            for line in sub_dump:
                comment = json.loads(line)
                comment_link_id = comment['link_id']
                comment_author = comment['author']
                try:
                    authors_subs[comment_link_id].append(comment_author)
                except KeyError:
                    authors_subs[comment_link_id] = []
                    authors_subs[comment_link_id].append(comment_author)
        total[subfile] = authors_subs
        i+=1
    return total        
        
#%%

YEAR = 2009
MONTH = ['01','02','03','04','05','06','07']

for m in MONTH:   
    print(m)
    authors = get_authors_all(YEAR, m)
    os.chdir('..')
    os.chdir('..')
    with open('authors_'+str(YEAR)+'-'+str(m) + '.pkl', 'wb') as f:
            pickle.dump(authors, f, pickle.HIGHEST_PROTOCOL)
    os.chdir('..')
