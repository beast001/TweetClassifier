import streamlit as st
import pandas as pd
import re
import plost
from datetime import date
import string
import tweepy
from datetime import datetime, timedelta
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


#setting up twitter
consumerKey = 'S8iEJ1hMMBit7radc5FaPZkTQ'
consumerSecret = 'MvmrjGKV6TTbbAWpzqJtc6RuxyZEg9uwdYhdvWc5NEiucn2Gh6'
accessToken = '1575957976090820619-vhfKRHgBKBPVS0Y7KbSmC1LqZzUNJk'
accessTokenSecret = 'I6tU33RIVclvJxZG5H4nhDWCDzZgSrq3Dpl88b2r5mBtO'

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
today = date.today() + timedelta(days=1)
#@st.cache(allow_output_mutation=True)
def getTweets(consumer_key, consumer_secret, access_token, access_token_secret, end_date =today,maxTweets=20):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Create API object
    api = tweepy.API(auth)

    username = 'safaricom_care'

    # Convert start_date and end_date to datetime objects
    #start_date = datetime.strptime(start_date, '%Y-%m-%d')
    #end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

    # Search for tweets directed to specified user within the specified date range
    tweets = tweepy.Cursor(api.search_tweets, q='to:'+username,  until=end_date).items(maxTweets)

    # Creating list to append tweet data to
    tweets_list = []

    # Print tweets
    for tweet in tweets:
        tweets_list.append([tweet.created_at, tweet.id, tweet.text, tweet.author.name, tweet.favorite_count,'https://twitter.com/anyuser/status/'+str(tweet.id)])
        
    # Creating a dataframe from the tweets list above
    #tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'ReplyCount', 'View'])
    global all_tweets, replied_tweets, unreplied_tweets
    all_tweets = len(tweets_list)
    replied_tweets = len(tweets_df[tweets_df['ReplyCount'] >= 1])
    unreplied_tweets = len(tweets_df[tweets_df['ReplyCount'] <= 0])
    tweets_df['Datetime'] = pd.to_datetime(tweets_df['Datetime'])
    tweets_df["hour"] = tweets_df["Datetime"].apply(lambda x: x.hour)

    
    # tweets_df.to_csv("tweets.csv", index = False)
    return tweets_df


st.set_page_config(page_title="Sentiment", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#sidebar   
st.sidebar.header('Departments `Safaricom_care`')

st.sidebar.subheader('Fetch Tweets') 
#tweete_from = str(st.sidebar.date_input("From",date.today(),max_value = date.today()))
tweete_to =st.sidebar.date_input("Fetch Tweets Until",date.today(),max_value = date.today()) + timedelta(days=1)
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 100, 5000, 20)

#st.sidebar.subheader('Select Department To Work On')
st.sidebar.subheader('Select Sentiment To Work On')
sentiment_select = st.sidebar.selectbox('Select Mood', ('All','Positive', 'Negative', 'Neutral'))


st.sidebar.markdown('''
---
Created with by [Team Alpha](https://github.com/beast001/TweetClassifier/).
''')
df = getTweets(consumerKey, consumerSecret, accessToken, accessTokenSecret, tweete_to,tweet_count)

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

st.markdown('### Sentiments')
#function to show number of unreplied tweets
def unreplied(df):
    a = len(df[df['ReplyCount'] <= 0])
    return f"{a} unreplied"
df_neg = unreplied(df[df['sentiment'] == "Negative emotion"])
df_pos = unreplied(df[df['sentiment'] == 'Positive emotion'])
df_neu = unreplied(df[df['sentiment'] == "Neutral emotion"])




col1, col2, col3 = st.columns(3)
col1.metric("Neutral Emotions", len(df[df['sentiment']=="Neutral emotion"]), df_neu)
col2.metric("Negative Emotions", len(df[df['sentiment']=="Negative emotion"]), df_neg )
col3.metric("Positive Emotions", len(df[df['sentiment']=="Positive emotion"]), df_pos)

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">View</a>'

# Column to view tweets with hyperlinks
df['Open'] = df['View'].apply(make_clickable)

with st.expander("ℹ️ How to interpret the results", expanded=False):
    st.write(
        """
        **Polarity**: Polarity is a float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement
        """
    )
    st.write("")

if sentiment_select == 'Positive':
    #st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('## Positive Sentiments')
    df_unique = df[df['sentiment'] == 'Positive emotion']
    df_unique = df_unique[df_unique['ReplyCount'] <= 0]
    st.write(df_unique[['Datetime', 'Text', 'Username','polarity','sentiment','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)

elif sentiment_select == 'Negative':
    st.markdown('## Negative Sentiments')
    df_unique = df[df['sentiment'] == "Negative emotion"]
    df_unique = df_unique[df_unique['ReplyCount'] <= 0]
    st.write(df_unique[['Datetime', 'Text', 'Username','polarity','sentiment','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)
elif sentiment_select == 'Neutral':
    st.markdown('## Neutral Sentiments')
    df_unique = df[df['sentiment'] == "Neutral emotion"]
    df_unique = df_unique[df_unique['ReplyCount'] <= 0]
    st.write(df_unique[['Datetime', 'Text', 'Username','polarity','sentiment','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)

else:
    st.markdown('## All Sentiments')
    st.write(df[['Datetime', 'Text', 'Username','polarity','sentiment','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)





