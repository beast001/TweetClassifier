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


st.set_page_config(page_title="Model", page_icon="📈",layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#sidebar   
st.sidebar.header('Departments `Safaricom_care`')

st.sidebar.subheader('Fetch Tweets') 
tweete_from = str(st.sidebar.date_input("From",date.today()))
tweete_to = str(st.sidebar.date_input("To",date.today()))
tweet_count = st.sidebar.slider('Specify Number Of Tweets', 200, 5000, 20)

st.sidebar.subheader('Select Department To Work On')
dept_select = st.sidebar.selectbox('Select data', ('General', 'Mpesa', 'Internet', 'Value Added Services', 'Voice', 'Customer Care', 'All Departments'))

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

# Row 1
st.markdown('### Metrics')
#row 1 columns
with st.container():
    left_column, center_column,right_column =  st.columns(3)
    with left_column:
        st.metric(label="General", value=general)

        st.metric(label="Value Added Services", value=vservices)
        

    with center_column:
        st.metric(label="Mpesa", value=mpesa)

        st.metric(label="Customer Service", value=cservice)


    with right_column:
        st.metric(label="Internet", value=internet)

        st.metric(label="Voice", value=voice)




if dept_select == 'General':
    #st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('## General Department')
    AgGrid(df[df['Prediction'] == "General"])
elif dept_select == 'Mpesa':
    st.markdown('## Mpesa Depertments')
    AgGrid(df[df['Prediction'] == "Mpesa"])
    
elif dept_select == 'Internet':
    st.markdown('## Internet Depertments')
    AgGrid(df[df['Prediction'] == "Internet"])

elif dept_select == 'Value Added Services':
    st.markdown('## Value Added Services Depertments')
    AgGrid(df[df['Prediction'] == "Value added Service"])

elif dept_select == 'Voice':
    st.markdown('## Voice Depertments')
    AgGrid(df[df['Prediction'] == "Voice"])

elif dept_select == 'Customer Care':
    st.markdown('## Customer Care')
    AgGrid(df[df['Prediction'] == "Customer Care"])
else:
    st.markdown('## All Depertments') 
    AgGrid(df)




