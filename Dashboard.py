import streamlit as st
import pandas as pd
from pandas import Series
import re
import plost
from datetime import date
import string
import snscrape.modules.twitter as sntwitter
import seaborn as sns
import matplotlib.pyplot as plt

#Global Variables
all_tweets = 0
replied_tweets = 0
unreplied_tweets = 0
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
    tweets_df["hour"] = tweets_df["Datetime"].apply(lambda x: x.hour)

    
    # tweets_df.to_csv("tweets.csv", index = False)
    return tweets_df


st.set_page_config(page_title="Plotting Demo", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `Safaricom_care`')



st.sidebar.subheader('Fetch Tweets') 
tweete_from = str(st.sidebar.date_input("From",date.today()))
tweete_to = str(st.sidebar.date_input("To",date.today()))
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 200, 5000, 20)

#st.sidebar.subheader('Donut chart parameter')
#donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
#plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with by [Antony Brian](https://www.linkedin.com/in/africandatascientist/).
''')

# Row 1
df = getTweets(tweete_from, tweete_to, tweet_count)

st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("All Tweets Fetched", all_tweets)
col2.metric("Replied Tweets", replied_tweets, replied_tweets -unreplied_tweets)
col3.metric("Unreplied Tweets", unreplied_tweets)

#creating a new column with the clean tweets
clean_data=[]
for i in df["Text"]:
    clean_data.append(basic_clean(i))

df['Clean_Tweets']=clean_data
df['Time of Day'] = df['hour'].apply(get_time)

with st.container():
    left_column, right_column =  st.columns(2)
    with left_column:
        st.write("##")
        st.markdown('### Tweets distribution by time of day')
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x="Time of Day", data=df)
        st.pyplot(fig)

    with right_column:
        st.write("##")
        st.markdown('### Tweet Distribution Every Hour')
        def f(x):
            return Series(dict(Number_of_tweets = x['Text'].count(),))
        tweets_df = df.groupby(df.hour).apply(f)
        
        st.line_chart(tweets_df)

    
       
st.write(df[["Text", "Username","Time of Day","hour"]])
        









    
