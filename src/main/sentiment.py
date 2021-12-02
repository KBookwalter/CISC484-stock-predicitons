from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# In and out file name variables
in_file_name = "old_tesla_tweets.csv"
out_file_name = "tweets_with_sent.csv"

delim = ','

sia = SentimentIntensityAnalyzer()

# Read tweets
tweets = pd.read_csv(in_file_name, sep=',')


out_file = open(out_file_name, "w", encoding = "utf-8")

out_file.write("date, text, neg, neu, pos, compound")

for i in range(len(tweets)):
    scores = sia.polarity_scores(tweets['tweet'][i])
    out_file.write(tweets['date'][i] + delim)
    out_file.write(tweets['tweet'][i] + delim)
    out_file.write(str(scores["neg"]) + delim)
    out_file.write(str(scores["neu"]) + delim)
    out_file.write(str(scores["pos"]) + delim)
    out_file.write(str(scores["compound"]) + "\n")

out_file.close()
