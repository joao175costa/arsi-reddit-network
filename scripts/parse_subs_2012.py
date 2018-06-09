#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:24:08 2018

@author: togepi
"""

import os
import json

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'
SUBS_COM_PATH = PATH + 'subs_com/'
SUBS_POST_PATH = PATH + 'subs_post/'

com_file = PATH + 'RC_2012-10'
post_file = PATH + 'RS_2012-10'

#%%
i=0
with open(com_file) as jsonfile:
    for line in jsonfile:
        jsonline = json.loads(line)
        sub = jsonline['subreddit']   
        with open(SUBS_COM_PATH + sub, 'a') as subfile:
            subfile.write(json.dumps(jsonline)+'\n')
        i+=1
        if (i%1000 == 0):
            print(i)

#%%
i=0
with open(post_file) as jsonfile:
    for line in jsonfile:
        jsonline = json.loads(line)
        sub = jsonline['subreddit']  
        with open(SUBS_POST_PATH + sub, 'a') as subfile:
            subfile.write(json.dumps(jsonline)+'\n')
        i+=1
        if (i%1000 == 0):
            print(i)     
