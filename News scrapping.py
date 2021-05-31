#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:27:09 2021

@author: maryanneatakpa
"""

import pprint
import requests # 2.19.1
import requests
import json
import re
import pandas as pd
from pandas.io.json import json_normalize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#replace with your developer key from newapi.org
secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'

everything_news_url = 'https://newsapi.org/v2/everything'

# Specify the query and number of returns
parameters = {
 'q': 'dogecoin, cryptocurrency crypto market', # query phrase
 'pageSize': 30, # maximum is 100
 'apiKey': 'b93b15c8a9154b2db5fdbc83a303efb1' # your own API key
}

# Make the request
response = requests.get(everything_news_url, params=parameters)
# Convert the response to JSON format and pretty print it
response_json = response.json()

#build dataframe
datax = []
for m in response_json['articles']:
  datax.append(
        {
            'Title': m['title'],
            'Description': m['description'],
            'URL':m['url'],
            'Source':m['source']['name']
        }
    )

dfx = pd.DataFrame(datax)
# len(df)
dfx.head(15)

xz = dfx.iloc[0:1, 1]

print (xz)

import nltk
from nltk.corpus import stopwords

#sentiment analysis of Article titles

#load nlkt 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyser = SentimentIntensityAnalyzer()

#loop through Description and assign sentiment score
i=0
dfx_sentiment = dfx
#empty list to hold computed 'compound' VADER scores
compval1 = [ ]

while (i<len(dfx_sentiment)):

    k = analyser.polarity_scores(dfx_sentiment.iloc[i]['Description'])
    compval1.append(k['compound'])
    
    i = i+1

dfx_sentiment['Sentiment score'] = compval1
dfx_sentiment.head(15)

#Assigning score sentiment categories
i = 0

predicted_value = [ ] #empty series to hold our predicted values

while(i<len(dfx_sentiment)):
    if ((dfx_sentiment.iloc[i]['Sentiment score'] >= 0.2)):
        predicted_value.append('positive')
        i = i+1
    elif ((dfx_sentiment.iloc[i]['Sentiment score'] >= 0) & (dfx_sentiment.iloc[i]['Sentiment score'] < 0.7)):
        predicted_value.append('neutral')
        i = i+1
    elif ((dfx_sentiment.iloc[i]['Sentiment score'] <= 0)):
        predicted_value.append('negative')
        i = i+1

dfx_sentiment['sentiment'] = predicted_value

#Lets have a look at some good news in the midst of so much negative news 
positive = dfx_sentiment.loc[dfx_sentiment['sentiment'] == 'neutral']
positive


#Plot bar chart showing the sentiment levels
dfx_sentiment.groupby('sentiment').size().plot(kind='bar')

#word cloud to visualize most words in title

words = ' '.join(dfx['Title'])

wordcloud = WordCloud(background_color='white',
                      max_words=100,
                      width=1500,
                      height=1250
                     ).generate(words)

plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Create an empty string
text_combined = ''
# Loop through all the headlines and add them to 'text_combined'
for i in response_json['articles']:
 text_combined += i['title'] + ' ' # add a space after every headline, so the first and last words are not glued together
# Print the first 300 characters to screen for inspection
print(text_combined[0:300])

reuters= dfx[dfx['URL'].str.contains("reuters")]


#empty list to hold the scrapped text
reuters_article=[]

reutersurls =(reuters['URL']).to_list()

from bs4 import BeautifulSoup as sp
#looping through the urls to scrap the text
for l in reutersurls:
    qpage= requests.get(l)
    bsobjkq= sp(qpage.content)
    for news in bsobjkq.findAll('div',{'class':'ArticleBody__content___2gQno2 paywall-article'}):
      reuters_article.append(news.text.strip())

#we have extracted 9 articles from reuters


#extract 5 articles
n1 = reuters_article[1]
n2 = reuters_article[0]
n3 = reuters_article[2]
n4 = reuters_article[3]
n5 = reuters_article[4]

import requests
import urllib.request
import time
from bs4 import BeautifulSoup 
url1 = "https://www.reuters.com/article/us-crypto-currency-musk-idUSKBN2CO246"
page1 = requests.get(url1).text
# Turn page into BeautifulSoup object to access HTML tags
soup1 = BeautifulSoup(page1)

# Pares HTML for article body

# Get text from all <p> tags.
p_tags1 = sp.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.
p_tags_text1 = [tag.get_text().strip() for tag in p_tags1]

#convert to string for easier manipulation
text1 = " ".join([word for word in p_tags_text1
                            if '\xa0' not in word
                 ])

url2 = "https://www.reuters.com/article/us-crypto-currency-musk-idUSKBN2CO246"
page2 = requests.get(url2).text
# Turn page into BeautifulSoup object to access HTML tags
soup2 = BeautifulSoup(page2)

# Pares HTML for article body

# Get text from all <p> tags.
p_tags2 = sp.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.
p_tags_text2 = [tag.get_text().strip() for tag in p_tags2]

#convert to string for easier manipulation
text2 = " ".join([word for word in p_tags_text2
                            if '\xa0' not in word
                 ])

url3 = "https://www.reuters.com/article/us-crypto-currency-musk-copy-idUKKBN2CO271"
page3 = requests.get(url3).text
# Turn page into BeautifulSoup object to access HTML tags
soup3 = BeautifulSoup(page3)

# Pares HTML for article body

# Get text from all <p> tags.
p_tags3 = sp.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.https://www.reuters.com/article/us-crypto-currency-musk-copy-idUKKBN2CO271
p_tags_text3 = [tag.get_text().strip() for tag in p_tags3]

#convert to string for easier manipulation
text3 = " ".join([word for word in p_tags_text3
                            if '\xa0' not in word
                 ])


url4 = "https://www.reuters.com/technology/meme-based-cryptocurrency-dogecoin-soars-40-all-time-high-2021-05-05/"
page4 = requests.get(url4).text
# Turn page into BeautifulSoup object to access HTML tags
soup4 = BeautifulSoup(page4)

# Pares HTML for article body

# Get text from all <p> tags.
p_tags4 = sp.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.
p_tags_text4 = [tag.get_text().strip() for tag in p_tags4]

#convert to string for easier manipulation
text4 = " ".join([word for word in p_tags_text4
                            if '\xa0' not in word
                 ])

url5 = "https://www.reuters.com/technology/dogeday-hashtags-help-meme-based-cryptocurrency-dogecoin-hit-new-high-2021-04-20/d"
page5 = requests.get(url5).text
# Turn page into BeautifulSoup object to access HTML tags
soup5 = BeautifulSoup(page5)

# Pares HTML for article body

# Get text from all <p> tags.
p_tags5 = sp.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.
p_tags_text5 = [tag.get_text().strip() for tag in p_tags5]
#convert to string for easier manipulation
text5 = " ".join([word for word in p_tags_text5
                            if '\xa0' not in word
                 ])

from pywsd.utils import lemmatize, lemmatize_sentence
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))  
stop_words.add('.')
stop_words.add('$')
stop_words.add('%')
stop_words.add('"')
stop_words.add('-')
stop_words.add('#')    
stop_words.add('!')     
stop_words.add('(')    
stop_words.add(')')
stop_words.add('''''')
stop_words.add('')   
stop_words.add('1')    
stop_words.add("'") 
stop_words.add("''") 
stop_words.add(',') 
n1 =n1.lower()

 


#lemmatize and tokenize the words
ln1 = lemmatize_sentence(n1)

 

#clean the data by eliminating stopwords
filtered_n1=[w for w in ln1 if not w in stop_words]
freq_ln1 =nltk.FreqDist(filtered_n1)

 

#Eliminating words with three characters and below
large_ln1=dict([(k,v) for k,v in freq_ln1.items() if len(k) >3])


freq_ln1.plot(30, cumulative= False)

n1_cloud= WordCloud(max_font_size =50, max_words=100, background_color="white").generate_from_frequencies(freq_ln1)

plt.figure()
plt.imshow(n1_cloud, interpolation ="bilinear")
plt.axis("off")
plt.show()


n2 =n2.lower()

 


#lemmatize and tokenize the words
ln2 = lemmatize_sentence(n2)

 

#clean the data by eliminating stopwords
filtered_n2=[w for w in ln2 if not w in stop_words]
freq_ln2 =nltk.FreqDist(filtered_n2)

 

#Eliminating words with three characters and below
large_ln2=dict([(k,v) for k,v in freq_ln2.items() if len(k) >3])


freq_ln2.plot(30, cumulative= False)

n2_cloud= WordCloud(max_font_size =50, max_words=100, background_color="white").generate_from_frequencies(freq_ln2)

plt.figure()
plt.imshow(n2_cloud, interpolation ="bilinear")
plt.axis("off")
plt.show()

n3 =n3.lower()

 


#lemmatize and tokenize the words
ln3 = lemmatize_sentence(n3)

 

#clean the data by eliminating stopwords
filtered_n3=[w for w in ln3 if not w in stop_words]
freq_ln3 =nltk.FreqDist(filtered_n3)

 

#Eliminating words with three characters and below
large_ln3=dict([(k,v) for k,v in freq_ln3.items() if len(k) >3])


freq_ln3.plot(30, cumulative= False)

n3_cloud= WordCloud(max_font_size =50, max_words=100, background_color="white").generate_from_frequencies(freq_ln3)

plt.figure()
plt.imshow(n3_cloud, interpolation ="bilinear")
plt.axis("off")
plt.show()

n4 =n4.lower()

 


#lemmatize and tokenize the words
ln4 = lemmatize_sentence(n4)

 

#clean the data by eliminating stopwords
filtered_n4=[w for w in ln3 if not w in stop_words]
freq_ln4 =nltk.FreqDist(filtered_n4)

 

#Eliminating words with three characters and below
large_ln4=dict([(k,v) for k,v in freq_ln4.items() if len(k) >3])


freq_ln4.plot(30, cumulative= False)

n4_cloud= WordCloud(max_font_size =50, max_words=100, background_color="white").generate_from_frequencies(freq_ln4)

plt.figure()
plt.imshow(n4_cloud, interpolation ="bilinear")
plt.axis("off")
plt.show()

n5 =n5.lower()

 


#lemmatize and tokenize the words
ln5 = lemmatize_sentence(n5)

 

#clean the data by eliminating stopwords
filtered_n5=[w for w in ln3 if not w in stop_words]
freq_ln5 =nltk.FreqDist(filtered_n5)

 

#Eliminating words with three characters and below
large_ln5=dict([(k,v) for k,v in freq_ln5.items() if len(k) >3])


freq_ln5.plot(30, cumulative= False)

n5_cloud= WordCloud(max_font_size =50, max_words=100, background_color="white").generate_from_frequencies(freq_ln5)

plt.figure()
plt.imshow(n4_cloud, interpolation ="bilinear")
plt.axis("off")
plt.show()

import sklearn;
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer;
from sklearn.decomposition import LatentDirichletAllocation

#display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
     print ("Topic", topic_idx)
     print (" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# LDA is able to use tf-idf
no_features = 5000
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(filtered_n1, y=None)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#Initialize the number of Topics we need to cluster:
num_topics = 10;


lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)





no_top_words = 6
display_topics(lda, tfidf_feature_names, no_top_words)



#display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
     print ("Topic", topic_idx)
     print (" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# LDA is able to use tf-idf
no_features = 5000
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(filtered_n2, y=None)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#Initialize the number of Topics we need to cluster:
num_topics = 10;


lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)





no_top_words = 6
display_topics(lda, tfidf_feature_names, no_top_words)

#display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
     print ("Topic", topic_idx)
     print (" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# LDA is able to use tf-idf
no_features = 5000
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(filtered_n3, y=None)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#Initialize the number of Topics we need to cluster:
num_topics = 10;


lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)





no_top_words = 6
display_topics(lda, tfidf_feature_names, no_top_words)

#display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
     print ("Topic", topic_idx)
     print (" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# LDA is able to use tf-idf
no_features = 5000
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(filtered_n4, y=None)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#Initialize the number of Topics we need to cluster:
num_topics = 10;


lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)





no_top_words = 6
display_topics(lda, tfidf_feature_names, no_top_words)

#display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
     print ("Topic", topic_idx)
     print (" ".join([feature_names[i]
        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# LDA is able to use tf-idf
no_features = 5000
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=1, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(filtered_n5, y=None)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#Initialize the number of Topics we need to cluster:
num_topics = 10;


lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf)





no_top_words = 6
display_topics(lda, tfidf_feature_names, no_top_words)


n1

#summarize text

from gensim.summarization.summarizer import summarize
print(summarize(n1, word_count= 300))













