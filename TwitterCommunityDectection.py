#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:37:24 2021

@author: maryanneatakpa
"""
import tweepy 
import pandas as pd

consumer_key='xxxxxxxxxxxxxxxxx'                               
consumer_secret_key='xxxxxxxxxxxxxxxxx'  
access_token='xxxxxxxxxxxxxxx'  
access_token_secret='xxxxxxxxxxxxxx'  

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)         
auth.set_access_token(access_token, access_token_secret)     
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)        

me = api.get_user(screen_name = 'the_annea')      
me.id 

user_list = ["1404718962"]  
follower_list = []
for user in user_list:
    followers = []
    try:
        for page in tweepy.Cursor(api.followers_ids, user_id=user).pages():
            followers.extend(page)
            print(len(followers))
    except tweepy.TweepError:
        print("error")
        continue
    follower_list.append(followers)                          
    
df = pd.DataFrame(columns=['source','target']) #Empty DataFrame
df['target'] = follower_list[0] #Set the list of followers as the target column
df['source'] = 1404718962 #Set my user ID as the source 

import networkx as nx
G = nx.from_pandas_edgelist(df, 'source', 'target') #Turn df into graph
pos = nx.spring_layout(G) #specify layout for visual

import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')
nodes = nx.draw_networkx_nodes(G, pos,
                               alpha=0.8)
nodes.set_edgecolor('k')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.2)

user_list = list(df['target']) #Use the list of followers we extracted in the code above i.e. my 450 followers
for userID in user_list:
    print(userID)
    followers = []
    follower_list = []

    # fetching the user
    user = api.get_user(userID)

    # fetching the followers_count
    followers_count = user.followers_count

    try:
        for page in tweepy.Cursor(api.followers_ids, user_id=userID).pages():
            followers.extend(page)
            print(len(followers))
            if followers_count >= 10: #Only take first 5000 followers
                break
    except tweepy.TweepError:
        print("error")
        continue
    follower_list.append(followers)
    temp = pd.DataFrame(columns=['source', 'target'])
    temp['target'] = follower_list[0]
    temp['source'] = userID
    df = df.append(temp)
    df.to_csv("networkOfFollowers.csv")
    
tf = pd.read_csv('networkOfFollowers.csv')

gtf = nx.from_pandas_edgelist(tf,'source','target')

gtf.number_of_nodes() #Find the total number of nodes in this graph

g_sorted = pd.DataFrame(sorted(gtf.degree,key = lambda x:x[1],reverse = True))

g_sorted.columns = ['names', 'degree']

g_sorted.head

x = api.get_user(1901298962)

x.screen_name

g_r = nx.k_core(gtf, 4)

g_rf = nx.to_pandas_edgelist(g_r)

G = nx.from_pandas_edgelist(g_rf, source = 'source',target = 'target',create_using=nx.Graph() )

G.number_of_nodes()

nx.info(G)

pos = nx.spring_layout(G)

f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')
nodes = nx.draw_networkx_nodes(G, pos,
                               alpha=0.8)
nodes.set_edgecolor('k')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.2)

from cdlib import algorithms, viz
import networkx as nx

coms = algorithms.louvain(G, weight = "weight", resolution=1.)

viz.plot_network_clusters(G,coms,pos)
viz.plot_community_graph(G,coms)





