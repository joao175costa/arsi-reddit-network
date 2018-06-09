#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:53:25 2018

@author: togepi
"""

import pickle

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'

with open(PATH + 'filt_com_dict.pkl', 'rb') as f:
    coms = pickle.load(f)

print('loaded')
#%%
comments = []
subs = []
stamps = []
for sub in coms:
    for post in coms[sub]:
        for c in coms[sub][post]:
            text = coms[sub][post][c]['body']
            ts = coms[sub][post][c]['created_utc']
            if (text != '[deleted]'):
                comments.append(text)
                subs.append(sub)
                stamps.append(ts)

print('vectors created')

#%%
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(comments, subs)
print('split')

#%%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

def preprocess_comment(comment):
    tokens = word_tokenize(comment)
    tokens_lower = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)
    stripped_tokens = [w.translate(table) for w in tokens_lower]
    words = [word for word in stripped_tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words_no_stop = [w for w in words if not w in stop_words]
    
    words_no_short = [w for w in words_no_stop if len(w)>=3]
        
    return words_no_short
    
def stem_comment(comment):
    snowball = SnowballStemmer('english')
    stemmed = [snowball.stem(word) for word in comment]
    return stemmed

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english', 
                       preprocessor = preprocess_comment,
                       tokenizer = stem_comment,
                       max_features = 5000)

dtm = vect.fit_transform(Xtrain)
words = vect.get_feature_names()
print('dtm matrix')

from sklearn.feature_selection import SelectPercentile, mutual_info_classif

selector = SelectPercentile(mutual_info_classif, percentile=20)
dtm_reduced = selector.fit_transform(dtm, ytrain)
selector_scores = selector.scores_
print('selected')

dtm_test = vect.transform(Xtest)
dtm_selected = selector.transform(dtm_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

model_random_forest = RandomForestClassifier()
model_random_forest.fit(dtm_reduced, ytrain)
prob_pred = model_random_forest.predict_proba(dtm_selected)
pred = model_random_forest.predict(dtm_selected)

cm_random_forest = confusion_matrix(ytest, pred)
cr_random_forest = classification_report(ytest, pred)

#%%
print('performing sentiment analysis')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
for sub in coms:
    for post in coms[sub]:
        for c in coms[sub][post]:
            text = coms[sub][post][c]['body']
            sents = sid.polarity_scores(text)
            # if text is deleted, score is neutral
            coms[sub][post][c]['neg'] = sents['neg']
            coms[sub][post][c]['neu'] = sents['neu']
            coms[sub][post][c]['pos'] = sents['pos']
            coms[sub][post][c]['compound'] = sents['compound']
print('done')

#%%
DEM_DICT = ['obama', 'biden', 'liber', 'barack', 'joe', 'democrat',
            'dem', 'libtard','obamacar', 'left', 'lefti', 'leftist']
REP_DICT = ['republican', 'republ', 'conserv', 'conservat',
            'gop', 'mitt', 'romney', 'paul', 'ryan', 'right']

for sub in coms:
    for post in coms[sub]:
        for c in coms[sub][post]:
            text = coms[sub][post][c]['body']
            stemmed_tokens = stem_comment(preprocess_comment(text))
            coms[sub][post][c]['about_dem']=False
            coms[sub][post][c]['about_rep']=False
            for token in stemmed_tokens:
                if token in DEM_DICT:
                    coms[sub][post][c]['about_dem']=True
                if token in REP_DICT:
                    coms[sub][post][c]['about_rep']=True

#%%
sub_sentiment = {}
for sub in coms:
    dem_pos = 0
    dem_neg = 0
    rep_pos = 0
    rep_neg = 0
    for post in coms[sub]:
        for c in coms[sub][post]:
            if (coms[sub][post][c]['about_dem'] and not coms[sub][post][c]['about_rep']):
                dem_pos += coms[sub][post][c]['pos'] * coms[sub][post][c]['ups']
                dem_neg += coms[sub][post][c]['neg'] * coms[sub][post][c]['ups']
            if (coms[sub][post][c]['about_rep'] and not coms[sub][post][c]['about_dem']):
                rep_pos += coms[sub][post][c]['pos'] * coms[sub][post][c]['ups']
                rep_neg += coms[sub][post][c]['neg'] * coms[sub][post][c]['ups'] 
    sub_sentiment[sub] = [dem_pos, dem_neg, rep_pos, rep_neg]                   
    print(sub, dem_pos, dem_neg, rep_pos, rep_neg)

#%%
import numpy as np

data_subs = {}
for sub in coms:
    data_subs[sub]={}
    data_subs[sub]['dem']=[]
    data_subs[sub]['rep']=[]
    print(sub)
    for post in coms[sub]:
        for c in coms[sub][post]:
            ts = coms[sub][post][c]['created_utc']
            pos = coms[sub][post][c]['pos']
            neu = coms[sub][post][c]['neu']
            neg = coms[sub][post][c]['neg']
            com = coms[sub][post][c]['compound']
            ups = coms[sub][post][c]['ups']
            about_dem = coms[sub][post][c]['about_dem']
            about_rep = coms[sub][post][c]['about_rep']
            if (about_dem and not about_rep):
                data_subs[sub]['dem'].append([ts, pos, neu, neg, com, ups])
            if (about_rep and not about_dem):
                data_subs[sub]['rep'].append([ts, pos, neu, neg, com, ups])
    data_subs[sub]['dem'] = np.array(data_subs[sub]['dem'])
    data_subs[sub]['rep'] = np.array(data_subs[sub]['rep'])
    
print('saving')
with open(PATH + 'graph_data.pkl', 'wb') as f:
    pickle.dump(data_subs, f, -1)
    
#%%

import pickle

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'
    
with open(PATH + 'graph_data.pkl', 'rb') as f:
    data_subs = pickle.load(f)
#%%
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime
import pytz
from pytz import timezone

EDT = timezone('America/Detroit')
start_month = datetime.fromtimestamp(1349049600+4*3600).astimezone(EDT)
end_month   = datetime.fromtimestamp(1351727999+4*3600).astimezone(EDT)
debate_03   = datetime.fromtimestamp(1349298000+4*3600).astimezone(EDT)
debate_11   = datetime.fromtimestamp(1349989200+4*3600).astimezone(EDT)
debate_16_s = datetime.fromtimestamp(1350421200+4*3600).astimezone(EDT)
debate_16_e = datetime.fromtimestamp(1350427200+4*3600).astimezone(EDT)
debate_22   = datetime.fromtimestamp(1350939600+4*3600).astimezone(EDT)

subreddit = 'Conservative'

politics_dem = data_subs[subreddit]['dem']
politics_rep = data_subs[subreddit]['rep']

def get_vectors(politics_dem):
    timestamps = np.array([int(ts) for ts in politics_dem[:,0]])
    sentiment = np.array([float(s) for s in politics_dem[:,4]])
    ups = np.array([int(up) for up in politics_dem[:,-1]])
    
    politics_dem = np.vstack((timestamps,sentiment,ups)).T
    politics_dem = politics_dem[politics_dem[:,0].argsort()]
    
    timestamps = politics_dem[:,0]
    sentiment = politics_dem[:,1]
    ups = politics_dem[:,2]
    return timestamps, sentiment, ups

#plt.plot(counts)

def moving_average(ts, sent, ups):
    filt_ts = []
    filt_sent = []
    max_ts = 1351728000
    min_ts = 1349049600+4*3600
    int_size = 3600#1 hour
    for i in range(31*24-1):
        low = min_ts + i*int_size
        high = min_ts + (i+1)*int_size
        
        index = np.where(np.logical_and(ts>low,ts<high))
        avg_ts = (low+high)/2
        weighted_scores = sent[index]*ups[index]
        weighted_pos = 0
        weighted_neg = 0
        for i in range(len(weighted_scores)):
            if (weighted_scores[i]>=0):
                weighted_pos += weighted_scores[i]
            else:
                weighted_neg += weighted_scores[i]
        try:
            #print(weighted_pos, weighted_neg)
            weighted_pos = np.abs(weighted_pos)
            weighted_neg = np.abs(weighted_neg)
            avg_sent = 2*(weighted_pos/(weighted_pos+weighted_neg))-1
            if np.isnan(avg_sent):
                raise ZeroDivisionError
            filt_ts.append(avg_ts)
            filt_sent.append(avg_sent)
        except ZeroDivisionError:
            pass
            
    filter_size = 24
    filt_ts = np.array(filt_ts)
    filt_sent = np.array(filt_sent)
    filt_ts = np.convolve(filt_ts, np.ones(filter_size)/filter_size, mode='valid')
    filt_sent = np.convolve(filt_sent, np.ones(filter_size)/filter_size, mode='valid')
    return filt_ts, filt_sent

t_dem, s_dem, u_dem = get_vectors(politics_dem)
t_rep, s_rep, u_rep = get_vectors(politics_rep)
fts_dem, fsent_dem = moving_average(t_dem, s_dem, u_dem)
fts_rep, fsent_rep = moving_average(t_rep, s_rep, u_rep)

dates_dem = [datetime.fromtimestamp(t).astimezone(EDT) for t in fts_dem]
dates_rep = [datetime.fromtimestamp(t).astimezone(EDT) for t in fts_rep]
plt.figure()
min_y = min(min(fsent_dem), min(fsent_rep))-.1
max_y = max(max(fsent_dem), max(fsent_rep))+.1
plt.plot(dates_dem, fsent_dem, 'b', label = 'Democrat')
plt.plot(dates_rep, fsent_rep, 'r', label = 'Republican')
plt.plot([debate_03, debate_03], [min_y, max_y], 'g:')
plt.plot([debate_11, debate_11], [min_y, max_y], 'g:')
plt.plot([debate_16_s, debate_16_s], [min_y, max_y], 'g:')
plt.plot([debate_22, debate_22], [min_y, max_y], 'g:')
plt.axis([start_month, end_month, min_y, max_y])
plt.title('Comment Sentiment of r/'+subreddit+' during October 2012')
plt.xlabel('Date')
plt.ylabel('Comment Sentiment Score')
plt.legend()

#%%
counts_dem, ints_dem = np.histogram(t_dem, bins = [start_month.timestamp()+i*3600 for i in range(31*24)])
counts_rep, ints_rep = np.histogram(t_rep, bins = [start_month.timestamp()+i*3600 for i in range(31*24)])
ints_dem_avg = (ints_dem[1:]+ints_dem[:-1])/2
dates_dem = [datetime.fromtimestamp(ints).astimezone(EDT) for ints in ints_dem_avg]
ints_rep_avg = (ints_rep[1:]+ints_rep[:-1])/2
dates_rep = [datetime.fromtimestamp(ints).astimezone(EDT) for ints in ints_rep_avg]
min_y = 0
max_y = max(max(counts_dem), max(counts_rep))+100
plt.figure()
plt.plot(dates_dem, counts_dem, 'b', label = 'Democrat')
plt.plot(dates_rep, counts_rep, 'r', label = 'Republican')
plt.plot([debate_03, debate_03], [min_y, max_y], 'g:')
plt.plot([debate_11, debate_11], [min_y, max_y], 'g:')
plt.plot([debate_16_s, debate_16_s], [min_y, max_y], 'g:')
plt.plot([debate_22, debate_22], [min_y, max_y], 'g:')
plt.axis([start_month, end_month, min_y, max_y])
plt.title('Comment Activity of r/' + subreddit + ' during October 2012')
plt.xlabel('Date')
plt.ylabel('# of Comments')
plt.legend()

#%%
start_debate = 1350435600 - 1*3600
duration = 2*3600
end_debate = 1350435600 + duration + 2*3600

debate_dem_index = np.where(np.logical_and(t_dem>start_debate, t_dem<end_debate))
debate_rep_index = np.where(np.logical_and(t_rep>start_debate, t_rep<end_debate))

debate_ts_dem = t_dem[debate_dem_index]
debate_s_dem = s_dem[debate_dem_index]
debate_u_dem = u_dem[debate_dem_index]

debate_ts_rep = t_rep[debate_rep_index]
debate_s_rep = s_rep[debate_rep_index]
debate_u_rep = u_rep[debate_rep_index]

def debate_moving_average(ts, sent, ups, start, end):
    filt_ts = []
    filt_sent = []
    min_ts = start
    max_ts = end
    n_hours = int((max_ts-min_ts)/3600)
    int_size = 60#1 minute
    for i in range(n_hours*60-1):
        low = min_ts + i*int_size
        high = min_ts + (i+1)*int_size
        
        index = np.where(np.logical_and(ts>low,ts<high))
        avg_ts = (low+high)/2
        weighted_scores = sent[index]*ups[index]
        weighted_pos = 0
        weighted_neg = 0
        for i in range(len(weighted_scores)):
            if (weighted_scores[i]>=0):
                weighted_pos += weighted_scores[i]
            else:
                weighted_neg += weighted_scores[i]
        try:
            #print(weighted_pos, weighted_neg)
            weighted_pos = np.abs(weighted_pos)
            weighted_neg = np.abs(weighted_neg)
            avg_sent = 2*(weighted_pos/(weighted_pos+weighted_neg))-1
            if np.isnan(avg_sent):
                raise ZeroDivisionError
            filt_ts.append(avg_ts)
            filt_sent.append(avg_sent)
        except ZeroDivisionError:
            pass
            
    filter_size = 15
    filt_ts = np.array(filt_ts)
    filt_sent = np.array(filt_sent)
    filt_ts = np.convolve(filt_ts, np.ones(filter_size)/filter_size, mode='valid')
    filt_sent = np.convolve(filt_sent, np.ones(filter_size)/filter_size, mode='valid')
    return filt_ts, filt_sent

deb_fts_dem, deb_fsent_dem = debate_moving_average(debate_ts_dem, debate_s_dem, debate_u_dem, 
                                                   start=start_debate, end=end_debate)
deb_fts_rep, deb_fsent_rep = debate_moving_average(debate_ts_rep, debate_s_rep, debate_u_rep,
                                                   start=start_debate, end=end_debate)

debate_dates_dem = [datetime.fromtimestamp(t).astimezone(EDT) for t in deb_fts_dem]
debate_dates_rep = [datetime.fromtimestamp(t).astimezone(EDT) for t in deb_fts_rep]
min_y = min(min(deb_fsent_dem), min(deb_fsent_rep))-.1
max_y = max(max(deb_fsent_dem), max(deb_fsent_rep))+.1
plt.figure()
plt.plot(debate_dates_dem, deb_fsent_dem, 'b', label = 'Democrat')
plt.plot(debate_dates_rep, deb_fsent_rep, 'r', label = 'Republican')
plt.plot([debate_16_s, debate_16_s], [min_y, max_y], 'g:')
plt.plot([debate_16_e, debate_16_e], [min_y, max_y], 'g:')
plt.axis([datetime.fromtimestamp(start_debate).astimezone(EDT), 
          datetime.fromtimestamp(end_debate).astimezone(EDT),
          min_y, max_y])
plt.title('Comment Sentiment of r/'+subreddit+' during October 16th Presidential Debate')
plt.xlabel('Date')
plt.ylabel('Comment Sentiment Score')
plt.legend()
#%%
import pickle

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'
    
with open(PATH + 'filt_post_dict.pkl', 'rb') as f:
    com_dict = pickle.load(f)

pol_com = com_dict['politics']
#%%
com_stats = []
for post in pol_com:
    ts = pol_com[post]['created_utc']
    ups = pol_com[post]['num_comments']
    body = pol_com[post]['title']
    com_stats.append((ts,post, ups, body))
    
#%%
import numpy as np
com_stats.sort(key = lambda x: x[0])
ts_vec = np.array([int(c[0]) for c in com_stats])

#%%
from datetime import datetime
from pytz import timezone

EDT = timezone('US/Eastern')
start_month = datetime(2012,10,1,0,0,0, tzinfo = EDT)
end_month = datetime(2012,10,31,23,59,59, tzinfo = EDT)
interval = 3600*24

dict_best_coms = {}
for i in range(31):
    start_ts = start_month.timestamp() + i*interval
    end_ts = start_month.timestamp() + (i+1)*interval
    
    index = np.where(np.logical_and(ts_vec>start_ts, ts_vec<end_ts))[0]
    daily_coms = []
    for j in index:
        daily_coms.append(com_stats[j])
    daily_coms.sort(key = lambda x: -x[2])
    dict_best_coms[i+1] = daily_coms
    
#%%
import pickle

PATH = '/home/togepi/feup-projects/arsi-reddit-network/reddit_2012/'
SELECTED_POST = '11lj06'
SECOND_POST = '11gjq9'
with open(PATH + 'sent_com_dict.pkl', 'rb') as f:
    com_dict = pickle.load(f)
    
com_selected_post = com_dict['politics'][SECOND_POST]
with open(PATH + 'selected_post_coms.pkl', 'wb') as f:
    pickle.dump(com_selected_post, f, -1)
with open(PATH + 'filt_post_dict.pkl', 'rb') as f:
    post_dict = pickle.load(f)

#%%
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DEM_DICT = ['obama', 'biden', 'liber', 'barack', 'joe', 'democrat',
            'dem', 'libtard','obamacar', 'left', 'lefti', 'leftist']
REP_DICT = ['republican', 'republ', 'conserv', 'conservat',
            'gop', 'mitt', 'romney', 'paul', 'ryan', 'right']

post_info = post_dict['politics'][SECOND_POST]
com_selected_post = com_dict['politics'][SECOND_POST]
sid = SentimentIntensityAnalyzer()
text = post_info['title']
sent_scores = sid.polarity_scores(text)
post_info['compound'] = sent_scores['compound']
stemmed_tokens = stem_comment(preprocess_comment(text))
post_info['about_dem'] = False
post_info['about_rep'] = False
for token in stemmed_tokens:
    if token in DEM_DICT:
        post_info['about_dem']=True
    if token in REP_DICT:
        post_info['about_rep']=True

post_graph = nx.Graph()
post_graph.add_node(post_info['id'],
                    title = post_info['title'], 
                    ups = int(post_info['ups']), 
                    timestamp = int(post_info['created_utc']),
                    sentiment = float(post_info['compound']),
                    aboutdem = bool(post_info['about_dem']),
                    aboutrep = bool(['about_rep']))

for comment in com_selected_post:
    c_dict = com_selected_post[comment]
    text = c_dict['body']
    stemmed_tokens = stem_comment(preprocess_comment(text))
    c_dict['about_dem'] = False
    c_dict['about_rep'] = False
    for token in stemmed_tokens:
        if token in DEM_DICT:
            c_dict['about_dem']=True
        if token in REP_DICT:
            c_dict['about_rep']=True
    
    post_graph.add_node(c_dict['id'],
                        title = c_dict['body'],
                        ups = int(c_dict['ups']),
                        timestamp = int(c_dict['created_utc']),
                        sentiment = float(c_dict['compound']),
                        aboutdem = bool(c_dict['about_dem']),
                        aboutrep = bool(c_dict['about_rep']))
    
    post_graph.add_edge(c_dict['id'], c_dict['parent_id'][3:])

nx.readwrite.write_gexf(post_graph, PATH + 'post_graph_second.gexf')