#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:16:43 2018

@author: togepi
"""
from create_sub_graph import create_sub_gexf

year = [2007,2008,2009]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']

for y in year:
    for m in month:
        print(y,m)
        try:
            create_sub_gexf(y,m)
        except FileNotFoundError as e:
            print(e.strerror)