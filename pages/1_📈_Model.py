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


#setting up twitter
consumerKey = 'S8iEJ1hMMBit7radc5FaPZkTQ'
consumerSecret = 'MvmrjGKV6TTbbAWpzqJtc6RuxyZEg9uwdYhdvWc5NEiucn2Gh6'
accessToken = '1575957976090820619-vhfKRHgBKBPVS0Y7KbSmC1LqZzUNJk'
accessTokenSecret = 'I6tU33RIVclvJxZG5H4nhDWCDzZgSrq3Dpl88b2r5mBtO'


#Global Variables
all_tweets = 0
replied_tweets = 0
unreplied_tweets = 0

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
#@st.cache(allow_output_mutation=True)
def getTweets(consumer_key, consumer_secret, access_token, access_token_secret, start_date=today, end_date =today,maxTweets=20):
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
        tweets_list.append([tweet.created_at, tweet.id, tweet.text, tweet.author.name, 0,'https://twitter.com/anyuser/status/'+str(tweet.id)])
        
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

st.set_page_config(page_title="Model", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#sidebar   
st.sidebar.header('Departments `Safaricom_care`')

st.sidebar.subheader('Fetch Tweets') 
tweete_from = str(st.sidebar.date_input("From",date.today(),max_value = date.today()))
tweete_to = str(st.sidebar.date_input("To",date.today(),max_value = date.today()))
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 200, 5000, 20)

st.sidebar.subheader('Select Department To Work On')
dept_select = st.sidebar.selectbox('Select data', ('General', 'Mpesa', 'Internet', 'Value Added Services', 'Voice', 'Customer Care', 'All Departments'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
#plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with by [Team Alpha](https://github.com/beast001/TweetClassifier/).
''')
df = getTweets(consumerKey, consumerSecret, accessToken, accessTokenSecret, tweete_from, tweete_to,tweet_count)

#creating clean tweets
clean_data=[]
for i in df["Text"]:
    clean_data.append(basic_clean(i))

#loading and running the model
filename = 'saf_model.sav'
loaded_model = joblib.load(filename)
result = loaded_model.predict(clean_data)
depts = {'General':0,
        'Mpesa':0,
        'Internet':0,
        'Value added Service':0,
        'Voice':0,
        'Customer Care':0    
       }
for x in result:
    if x == 1:
        depts['General'] +=1
        
    elif x == 2:
        depts['Internet'] +=1
    elif x == 3:
        depts['Mpesa'] +=1
    elif x == 4:
        depts['Value added Service'] +=1
    elif x == 5:
        depts['Voice'] +=1
    else:
        depts['Customer Care'] +=1

mpesa = depts['Mpesa']
general = depts['General']
internet = depts['Internet']
vservices = depts['Value added Service']
voice = depts['Voice']
cservice = depts['Customer Care']

#adding prediction row to df
df['Prediction']= pd.DataFrame(result)
df.Prediction.replace({1:"General"},inplace=True)
df.Prediction.replace({2:"Internet"},inplace=True)
df.Prediction.replace({3:"Mpesa"},inplace=True)
df.Prediction.replace({4:"Value added Service"},inplace=True)
df.Prediction.replace({5:"Voice"},inplace=True)
df.Prediction.replace({0:"Customer Care"},inplace=True)

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">View</a>'

# Column to view tweets with hyperlinks
df['Open'] = df['View'].apply(make_clickable)

# Row 1
st.markdown('### Predictions for all Unreplied Tweets')
#row 1 columns
with st.container():
    left_column, center_column,right_column =  st.columns(3)
    with left_column:
        st.metric(label="General", value=general)        

    with center_column:
        st.metric(label="Mpesa", value=mpesa)

    with right_column:
        st.metric(label="Internet", value=internet)

    left_column, center_column,right_column =  st.columns(3)
    with left_column:
        st.metric(label="Value Added Services", value=vservices)
        

    with center_column:
        st.metric(label="Customer Service", value=cservice)


    with right_column:
        st.metric(label="Voice", value=voice)





if dept_select == 'General':
    #st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('## General Department')
    df_unique = df[df['Prediction'] == "General"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open']].to_html(escape=False, index=False), unsafe_allow_html=True)

elif dept_select == 'Mpesa':
    st.markdown('## Mpesa Department')
    df_unique = df[df['Prediction'] == "Mpesa"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)
    
elif dept_select == 'Internet':
    st.markdown('## Internet Department')
    df_unique = df[df['Prediction'] == "Internet"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)

elif dept_select == 'Value Added Services':
    st.markdown('## Value Added Services Department')
    df_unique = df[df['Prediction'] == "Value added Service"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)

elif dept_select == 'Voice':
    st.markdown('## Voice Department')
    df_unique = df[df['Prediction'] == "Voice"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)

elif dept_select == 'Customer Care':
    st.markdown('## Customer Care')
    df_unique = df[df['Prediction'] == "Customer Care"]
    st.write(df_unique[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.markdown('## All Department')

    st.write(df[['Datetime', 'Text', 'Username','Open' ]].to_html(escape=False, index=False), unsafe_allow_html=True)








