#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:05:25 2018

@author: togepi
"""

import json

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'
SUBS_COM_PATH = PATH + 'subs_com/'
SUBS_POST_PATH = PATH + 'subs_post/'

SELECTED_SUBS = ['politics', 'PoliticalDiscussion', 'PoliticalHumor', 
                 'Conservative', 'Libertarian', 'Republican',
                 'democrats', 'obama', 'Romney']

#%%
dict_of_posts = {}
for sub in SELECTED_SUBS:
    dict_of_posts[sub] = {}
    post_counter = 0
    with open(SUBS_POST_PATH + sub) as subfile:
        for post in subfile:
            post_counter += 1
            post_dict = json.loads(post)
            post_id = post_dict['id']
            dict_of_posts[sub][post_id] = post_dict
    print(sub, post_counter)

import pickle
with open(PATH + 'post_dict.pkl', 'wb') as f:
    pickle.dump(dict_of_posts,f,-1)
#%%
comments_counter = {}
for sub in dict_of_posts:
    print(sub)
    n_comments = []
    for post in dict_of_posts[sub]:
        n_comments.append(dict_of_posts[sub][post]['num_comments'])
    comments_counter[sub] = n_comments


#%%
dict_of_com = {}

for sub in SELECTED_SUBS:
    dict_of_com[sub] = {}
    com_counter = 0
    with open(SUBS_COM_PATH + sub) as subfile:
        for com in subfile:
            com_counter += 1
            com_dict = json.loads(com)
            com_id = com_dict['name']
            post_id = com_dict['link_id'][3:]
            try:
                dict_of_com[sub][post_id][com_id] = com_dict
            except KeyError:
                dict_of_com[sub][post_id] = {}
                dict_of_com[sub][post_id][com_id] = com_dict
    print(sub, com_counter)

print('done')
import pickle
with open(PATH + 'com_dict.pkl', 'wb') as f:
    pickle.dump(dict_of_com,f,-1)
print('saved')

#%%
import pickle
def load_dicts():
    with open(PATH + 'post_dict.pkl', 'rb') as f:
        post_dict = pickle.load(f)
    with open(PATH + 'com_dict.pkl', 'rb') as f:
        com_dict = pickle.load(f)
    return post_dict, com_dict
    print('loaded')

posts, coms = load_dicts()

def filter_posts_coms(dict_of_posts, dict_of_coms, min_comments = 10):
    filtered_posts = {}
    for sub in dict_of_posts:
        filtered_posts[sub] = {}
        for post in dict_of_posts[sub]:
            n_comments = dict_of_posts[sub][post]['num_comments']
            if (n_comments > min_comments):
                filtered_posts[sub][post] = dict_of_posts[sub][post]
    print('posts filtered')
    
    filtered_coms = {}
    for sub in dict_of_coms:
        filtered_coms[sub] = {}
        for post in filtered_posts[sub]:
            try:
                for com in dict_of_coms[sub][post]:
                    try:
                        filtered_coms[sub][post][com] = dict_of_coms[sub][post][com]
                    except KeyError:
                        filtered_coms[sub][post] = {}
                        filtered_coms[sub][post][com] = dict_of_coms[sub][post][com]
            except KeyError:
                pass
        n_comments_sub_after = sum([len(filtered_coms[sub][p]) for p in filtered_coms[sub]])
        n_comments_sub_before = sum([len(dict_of_coms[sub][p]) for p in dict_of_coms[sub]])
        print(sub, '\nbefore:',n_comments_sub_before, ' after:', n_comments_sub_after)
    return filtered_posts, filtered_coms

posts, coms = filter_posts_coms(posts, coms)