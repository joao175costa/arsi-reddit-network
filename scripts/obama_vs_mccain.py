#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:54:21 2018

@author: togepi
"""

PATH = '/home/togepi/feup-projects/arsi-reddit-network/'

import pickle
import networkx as nx

def load_comments():
    with open(PATH + 'comments_politics_2008_10.pkl', 'rb') as f:
        comments = pickle.load(f)
    return comments

def load_links():
    with open(PATH + 'links_politics_2008_10.pkl', 'rb') as f:
        links = pickle.load(f)
    return links
    
def load_all():
    return load_comments(), load_links()

def add_links_to_graph(graph, links):
    for single_link in links:
        att = {'author': single_link['author'],
               'created_utc': single_link['created_utc'],
               'downs': single_link['downs'],
               'name': single_link['name'],
               'num_comments': single_link['num_comments'],
               'score': single_link['score'],
               'title': single_link['title'],
               'ups': single_link['ups']}
        graph.add_node(single_link['name'], isLink = True, **att)
        
def add_comments_to_graph(graph, comments):
    for single_comment in comments:
        att = {'author': single_comment['author'],
               'created_utc': single_comment['created_utc'],
               'downs': single_comment['downs'],
               'gilded': single_comment['gilded'],
               'link_id': single_comment['link_id'],
               'name': single_comment['name'],
               'parent_id': single_comment['parent_id'],
               'score': single_comment['score'],
               'body': single_comment['body'],
               'ups': single_comment['ups']}
        graph.add_node(single_comment['name'], isLink = False, **att)
        graph.add_edge(single_comment['name'], single_comment['link_id'])
        graph.add_edge(single_comment['name'], single_comment['parent_id'])
    
def add_all(graph, links, comments):
    add_links_to_graph(graph,links)
    add_comments_to_graph(graph,comments)
    
comm, links = load_all()
#%%
link_dict = {}
for l in links:
    link_dict[l['name']] = l

comm_dict = {}
for c in comm:
    comm_dict[c['name']] = c

#%%
link_comm_author_dict = {}
for comm in comm_dict:
    comm_in_link = comm_dict[comm]['link_id']
    comm_author = comm_dict[comm]['author']
    try:
        link_comm_author_dict[comm_in_link].append(comm_author)
    except:
        link_comm_author_dict[comm_in_link] = []
        link_comm_author_dict[comm_in_link].append(comm_author)
    
#%%
link_comm_unique_author_dict = {}
for link in link_comm_author_dict:
    unique_authors = set(link_comm_author_dict[link])
    try:
        unique_authors.remove('[deleted]')
    except:
        pass
    if (len(unique_authors) != 0):
        link_comm_unique_author_dict[link] = unique_authors
#%%
import numpy as np
n_links = len(link_comm_unique_author_dict)
shared_auths = np.zeros((n_links,n_links))
links_name = sorted(link_comm_unique_author_dict)
list_of_shared = []
for i in range(n_links):
    print(i)
    link_i = links_name[i]
    authors_i = link_comm_unique_author_dict[link_i]
    for j in range(i,n_links):
        link_j = links_name[j]
        authors_j = link_comm_unique_author_dict[link_j]
        shared = authors_i & authors_j
        n_shared = len(shared)
        shared_auths[i,j] = n_shared
        shared_auths[j,i] = n_shared
        if (n_shared != 0):
            list_of_shared.append(n_shared)
        
#%%
for i in range(len(shared_auths)):
    shared_auths[i,i] = 0

#%%
import numpy as np
list_of_shared = np.array(shared_authors)
#list_of_shared = 1/list_of_shared
hist = np.histogram(list_of_shared, bins=100)
log_count = np.log(hist[0])
log_int = np.log((hist[1][1:] + hist[1][0:-1])/2)
inf_mask = np.logical_not(np.isinf(log_count))
log_count = log_count[inf_mask]
log_int = log_int[inf_mask]

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model1.fit(log_int.reshape(-1,1), log_count)
r = model1.score(log_int.reshape(-1,1), log_count)
m1 = model1.coef_
print(r)
print(m1)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(log_int, model1.predict(log_int.reshape(-1,1)))
plt.legend(m1)
plt.plot(log_int,log_count,'o')
plt.xlabel('log(# of shared commentators)')
plt.ylabel('log(count)')
plt.title('Link distance distribution')

#%%
##%%
#import numpy as np
#ups = []
#n_comm = []
#ups_links = []
#for c in comm:
#    score = c['score']
#    if (score >= 0):
#        ups.append(score)
#ups = np.array(ups)
#
#for l in links:
#    n_comments = l['num_comments']
#    score = l['score']
#    if (score >= 0):
#        ups_links.append(score)
#    n_comm.append(n_comments)
#
##%%
#import matplotlib.pyplot as plt    
#
#counts_comm, intervals_comm = np.histogram(ups, bins = 100)
#log_counts_comm = np.log(counts_comm)
#intervals_comm = intervals_comm[1:]
#plt.plot(np.log(intervals_comm), log_counts_comm, 'o')
#
#counts_nr, intervals_nr = np.histogram(n_comm, bins = 100)
#log_counts_nr = np.log(counts_nr)
##plt.plot(np.log(intervals_nr[1:]), log_counts_nr, 'o')
#
#counts_ups_links, intervals_ups_links = np.histogram(ups_links, bins=100)
##plt.plot(np.log(intervals_ups_links[1:]), np.log(counts_ups_links), 'o')
##%%
#from sklearn.linear_model import LinearRegression
#inf_mask = np.logical_not(np.isinf(log_counts_comm))
#
#counts_noinf = np.log(counts_comm[inf_mask])
#intervals_comm = np.log(intervals_comm)
#intervals_noinf = intervals_comm[inf_mask]
#
#model1 = LinearRegression()
#model1.fit(intervals_noinf.reshape(-1,1), counts_noinf)
#model1.score(intervals_noinf.reshape(-1,1), counts_noinf)
#m1 = model1.coef_
#
#plt.figure()
#plt.plot(intervals_noinf, model1.predict(intervals_noinf.reshape(-1,1)))
#
#plt.title('m = %.2f' %m)