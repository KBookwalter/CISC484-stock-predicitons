import os
import datetime
from os import wait
from TwitterAPI import TwitterAPI
import tweepy as tw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

delim = ", "

##  To set tokens as environment variables, type in console:
##      export 'TOKEN_NAME'='TOKEN_VALUE'
##  replacing TOKEN_NAME and TOKEN_VALUE with respective values 
api_key = os.environ.get('API_KEY')
api_secret_key = os.environ.get('API_SECRET_KEY')
bearer_token = os.environ.get('BEARER_TOKEN')
access_token = os.environ.get('ACCESS_TOKEN')
access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')

auth = tw.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True)

search_word = "nhl"

date_since = datetime.date(2021, 11, 26)
date_until = datetime.date(2021, 1, 27)
num_days = 1

num_tweets = 1

def get_tweets(date):
    print(date)
    cursor = tw.Cursor(api.search_tweets, q=search_word, lang='en', since_id=date_since, count = num_tweets)
    tweets = cursor.items(num_tweets)
    return tweets

def get_sentiment(tweets):
    sum = 0
    for tweet in tweets:
        scores = sia.polarity_scores(tweet.text)
        sum += scores["compound"]
    return sum / num_tweets

date_list = [date_since + datetime.timedelta(days=x) for x in range(num_days)]

sents = []
for date in date_list:
    tweets = get_tweets(date)
    # sent = get_sentiment(tweets)
    # sents.append({"date": date, "sentiment": sent})
    for tweet in tweets:
        print(tweet.text)
        print(tweet.created_at)

for sent in sents:
    print(sent)

# for date in date_list:
#     print(date)

# out_file = open("twitter_sentiment.csv", "w", encoding="utf-8")
# out_file.write("")
# out_file.write("tweet_text, neg, neu, pos, compound\n")

# for tweet in tweets:
#     text = tweet.text
#     scores = sia.polarity_scores(text)
#     out_file.write(text.replace(",", "") + delim)
#     out_file.write(str(scores["neg"]) + delim)
#     out_file.write(str(scores["neu"]) + delim)
#     out_file.write(str(scores["pos"]) + delim)
#     out_file.write(str(scores["compound"]) + "\n")

# out_file.close()