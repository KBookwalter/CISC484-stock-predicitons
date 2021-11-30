import os
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

date_since = "2021-10-29"

num_tweets = 100

tweets = tw.Cursor(api.search_tweets, q=search_word, lang="en", since_id=date_since).items(num_tweets)

sum = 0

for tweet in tweets:
    scores = sia.polarity_scores(tweet.text)
    print(tweet.text)
    print(scores)
    print("\n")
    sum += scores["compound"]

print("Average Compound Score: ", sum/num_tweets)


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