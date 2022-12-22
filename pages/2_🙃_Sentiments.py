import streamlit as st
import pandas as pd
import re
import plost
from datetime import date
import string
import snscrape.modules.twitter as sntwitter
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from st_aggrid import GridOptionsBuilder, AgGrid, JsCode
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import numpy as np
from textblob import Word
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

#Global Variables
all_tweets = 0
replied_tweets = 0
unreplied_tweets = 0

PosE = 0
negE = 0
NeuE = 0

mpesa = 0
general = 0
internet = 0
vservices = 0
voice = 0
cservice = 0

#functions

# Defining functions to clean data
def lower_case(tweet):
    tweet = tweet.lower() 
    return tweet

#remove links
def remove_links(tweet):
    tweet = re.sub(r"https\S+"," ",tweet) #removes weblinks
    tweet = re.sub(r"bit.ly/\S+", " ",tweet) #removes weblinks
    return tweet

def remove_user(tweet):
    tweet = re.sub('(rt\s@[a-z]+[a-z0-9-_]+)', '',str(tweet)) #removes @user information
    tweet = re.sub('(@[a-z]+[a-z0-9-_]+)', '',str(tweet))#removes @user information
    return tweet

def remove_hashtags(tweet):
    tweeet = re.sub('(#[a-z]+[a-z0-9-_]+)', '',tweet) #removes the hashtags
    return tweet

def deEmojify(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def basic_clean(tweet):
    """Returns cleaned data, remove punctuation and numbers"""
    tweet = lower_case(tweet)
#     tweet = tokenization(tweet)
    tweet = remove_user(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = deEmojify(tweet)
#     tweet = remove_stopwords(tweet)
    tweet = re.sub('['+ string.punctuation+ ']+', '',tweet) # Removes punctuation
    tweet = re.sub('([0-9]+)'," ",tweet) # Removes numbers
    return tweet

#time to hour of day function
def get_time(hour):
    if hour >=6 and hour < 12:
         return 'Morning'
    if hour > 12 and hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'



#Getting todays tweets
today = str(date.today())
#@st.cache(allow_output_mutation=True)
def getTweets(sincedate=today, untildate=today, maxTweets=100):
    # Creating list to append tweet data to
    tweets_list = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Safaricom_Care since:2016-01-20').get_items()):
        if len(tweets_list)>maxTweets:
            break
        elif  tweet.username != "Safaricom_Care":
             tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username, tweet.replyCount,'https://twitter.com/anyuser/status/'+str(tweet.id)])
    # Creating a dataframe from the tweets list above
    #tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'ReplyCount', 'View'])
    global all_tweets, replied_tweets, unreplied_tweets
    all_tweets = len(tweets_list)
    replied_tweets = len(tweets_df[tweets_df['ReplyCount'] >= 1])
    unreplied_tweets = len(tweets_df[tweets_df['ReplyCount'] <= 0])
    tweets_df['Datetime'] = pd.to_datetime(tweets_df['Datetime'])


    
    # tweets_df.to_csv("tweets.csv", index = False)
    return tweets_df


st.set_page_config(page_title="Sentiment", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#sidebar   
st.sidebar.header('Departments `Safaricom_care`')

st.sidebar.subheader('Fetch Tweets') 
tweete_from = str(st.sidebar.date_input("From",date.today()))
tweete_to = str(st.sidebar.date_input("To",date.today()))
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 200, 5000, 20)

#st.sidebar.subheader('Select Department To Work On')
#dept_select = st.sidebar.selectbox('Select data', ('General', 'Mpesa', 'Internet', 'Value Added Services', 'Voice', 'Customer Care', 'All Departments'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
#plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with by [Antony Brian](https://www.linkedin.com/in/africandatascientist/).
''')
df = getTweets(tweete_from, tweete_to, tweet_count)
#creating clean tweets
clean_data=[]
for i in df["Text"]:
    clean_data.append(basic_clean(i))
df["Clean"] = clean_data

#st.write(tokenize(clean_data))
lemma = WordNetLemmatizer()
df['lemmatization'] = df.Clean.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
pol = []
for text in df["lemmatization"]:
    pol.append(np.sum(TextBlob(text).polarity))
df['polarity'] = pol

df.loc[(df["polarity"]>=0.4),"sentiment"] = "Positive emotion"
df.loc[(df["polarity"]<0),"sentiment"] = "Negative emotion"
df.loc[(df["polarity"].between(0,0.4,inclusive="left")),"sentiment"] = "Neutral emotion"

st.markdown('Sentiments')

col1, col2, col3 = st.columns(3)
col1.metric("Neutral Emotions", len(df[df['sentiment']=="Neutral emotion"]))
col2.metric("Negative Emotions", len(df[df['sentiment']=="Negative emotion"]) )
col3.metric("Positive Emotions", len(df[df['sentiment']=="Positive emotion"]))
st.write(df[["Text", "polarity","sentiment","View"]])





