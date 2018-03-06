#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:48:40 2018

@author: togepi
"""

def create_sub_gexf(YEAR, MONTH):
    
    import numpy as np
    import networkx as nx
    
    THRESHOLD =0
    
    #%%
    shared_authors = np.load(str(YEAR)+'/shared_authors_'+str(YEAR)+'-'+str(MONTH)+'.pkl')
    #shared_authors = shared_authors[:-1,:-1]
    subs = np.load(str(YEAR)+'/subs_'+str(YEAR)+'-'+str(MONTH)+'.pkl')
    #subs = subs[:-1]
    
    #%% COUNT COMMENTS FROM METAFILE
    from read_metadata import read_metafile
    
    path = str(YEAR)+'/meta/meta_RC_'+str(YEAR)+'-'+str(MONTH)
    sub_dict_comments = read_metafile(path)
    del sub_dict_comments['all']
    
    #%% COUNT UNIQUE AUTHORS FOR EACH SUB
    import pickle
    
    with open(str(YEAR)+"/authors_"+str(YEAR)+'-'+str(MONTH)+'.pkl', "rb") as input_file:
        sub_dict_authors_w_links = pickle.load(input_file)
    
    sub_dict_authors = {}
    for sub in sub_dict_authors_w_links:
        authors = []
        for thread in sub_dict_authors_w_links[sub]:
            auth_names = sub_dict_authors_w_links[sub][thread]
            for name in auth_names:
                authors.append(name)
        sub_dict_authors[sub]=len(set(authors))
    
    #%% SET NODE ATTRIBUTES
    list_of_subs = []
    for sub in subs:
        sub_name = sub
        nComments = sub_dict_comments[sub]
        nUniqueAuthors = sub_dict_authors[sub]
        att = {'comments':nComments, 'unique_authors':nUniqueAuthors}
        list_of_subs.append((sub_name, att))
    
    #%%
    graph = nx.Graph()
    graph.add_nodes_from(list_of_subs)
    
    
    connections = np.where(shared_authors > THRESHOLD)
    edges = []
    for i in range(np.shape(connections)[1]-1):
        #the -1 removes reddit.com
        sub1_i = connections[0][i]
        sub1 = subs[sub1_i]
        sub2_i = connections[1][i]
        sub2 = subs[sub2_i]
        w = shared_authors[sub1_i,sub2_i]
        edges.append((sub1, sub2, w))
    
    graph.add_weighted_edges_from(edges)
    
    #%%
    center_subs = 10
    shells = [[subs[-1]], subs[-center_subs:-1], subs[:-center_subs]]
    #nx.draw_shell(graph, nlist = shells, with_labels = True)
    options = {'edge_color': 'grey'}
    nx.write_gexf(graph,'gexf_subs_'+str(YEAR)+'-'+str(MONTH)+'.gexf')