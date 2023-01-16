import streamlit as st
import pandas as pd
from pandas import Series
import re
import joblib
import plost
from datetime import date
import string
import tweepy
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter
import seaborn as sns
import matplotlib.pyplot as plt


#setting up twitter
consumerKey = 'S8iEJ1hMMBit7radc5FaPZkTQ'
consumerSecret = 'MvmrjGKV6TTbbAWpzqJtc6RuxyZEg9uwdYhdvWc5NEiucn2Gh6'
accessToken = '1575957976090820619-vhfKRHgBKBPVS0Y7KbSmC1LqZzUNJk'
accessTokenSecret = 'I6tU33RIVclvJxZG5H4nhDWCDzZgSrq3Dpl88b2r5mBtO'

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
        return 'Night'



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


st.set_page_config(page_title="Plotting Demo", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `Safaricom_care`')



st.sidebar.subheader('Fetch Tweets') 
#tweete_from = str(st.sidebar.date_input("From",date.today(), max_value = date.today()))
tweete_to =st.sidebar.date_input("Fetch Tweets Until",date.today(),max_value = date.today()) + timedelta(days=1)
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 100, 5000, 20)

#st.sidebar.subheader('Donut chart parameter')
#donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
#plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with by [Team Alpha](https://github.com/beast001/TweetClassifier/).
''')

# Row 1
df = getTweets(consumerKey, consumerSecret, accessToken, accessTokenSecret, tweete_to,tweet_count)


st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("All Tweets Fetched", all_tweets)
col2.metric("Replied Tweets", replied_tweets)
col3.metric("Unreplied Tweets", unreplied_tweets)

#creating a new column with the clean tweets
clean_data=[]
for i in df["Text"]:
    clean_data.append(basic_clean(i))

#loading and running the model
filename = 'saf_model.sav'
loaded_model = joblib.load(filename)
result = loaded_model.predict(clean_data)

#adding prediction row to df
df['Prediction']= pd.DataFrame(result)
df.Prediction.replace({1:"General"},inplace=True)
df.Prediction.replace({2:"Internet"},inplace=True)
df.Prediction.replace({3:"Mpesa"},inplace=True)
df.Prediction.replace({4:"Value added Service"},inplace=True)
df.Prediction.replace({5:"Voice"},inplace=True)
df.Prediction.replace({0:"Customer Care"},inplace=True)

df['Clean_Tweets']=clean_data
df['Time of Day'] = df['hour'].apply(get_time)

with st.container():
    left_column, right_column =  st.columns(2)
    with left_column:
        st.write("##")
        st.markdown('### Tweets Distribution by Time of Day')
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(x="Time of Day", data=df)
        st.pyplot(fig)

    with right_column:
        st.write("##")
        st.markdown('### Tweet Distribution Every Hour')
        def f(x):
            return Series(dict(Number_of_tweets = x['Text'].count(),))
        tweets_df = df.groupby(df.hour).apply(f)
        
        st.line_chart(tweets_df)

    
with st.container():
    left_column2, right_column2 = st.columns(2)
    with left_column2:
        st.write('##')
        st.write("### Time of Day vs Category")
        fig3 = plt.figure(figsize=(10, 6))
        sns.set(style='white')
        sns.countplot(x='Time of Day', hue='Prediction', data=df)
        plt.xlabel('Time of Day')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig3)
              
        
    with right_column2:
        st.write('##')
        st.write("### Tweet Distribution")
        fig = plt.figure(figsize=(10, 6))
        sns.countplot(x="Prediction", data=df)
        st.pyplot(fig) 


        

#st.write("##")
#st.markdown("### Tweets Table")
#st.write(df[["Text", "Username","Time of Day","Prediction"]])
    
        
        
       









    
