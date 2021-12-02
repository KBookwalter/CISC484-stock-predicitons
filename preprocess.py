import csv
import re
tweets=[]
with open('old_tweets.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ')

    for line in csv_reader:
        if line !=  []:
            for token in line:
                if (token == '$tsla' or token  == '$TSLA'):
                    s=str(line[0])
                    basestring=''
                    if s != '':
                        if s[0].isdigit():
                            for c in line[2:]:
                                basestring=basestring+c+' '
                            basestring = re.sub(r'[^\w\s]','',basestring)
                            d=s.split(';')
                            newline=[d[1], basestring]
                            basestring=''
                            tweets.append(newline)

fields = ['date', 'tweet']

with open('old_tesla_tweets.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(fields)
    csvwriter.writerows(tweets)
