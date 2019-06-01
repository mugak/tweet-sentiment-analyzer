import nltk

geotagged_tweets, west_tweets, midwest_tweets, south_tweets, northeast_tweets = [], [], [], [], []
WEST = ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'AK', 'HI']
MIDWEST = ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'MI', 'IN', 'OH']
SOUTH = ['TX', 'OK', 'AR', 'LA', 'KY', 'TN', 'MS', 'AL', 'WV', 'WD', 'DE', 'DC', 'VA', 'NC', 'SC', 'GA', 'FL']
NORTHEAST = ['NY', 'PA', 'NJ', 'ME', 'VT', 'NH', 'MA', 'CT', 'RI']

for tweet in nltk.corpus.twitter_samples.docs():
    if tweet['place']:
        if tweet['lang'] == 'en' and tweet['place']['country_code'] == 'US':
            geotagged_tweets.append(tweet)

for tweet in geotagged_tweets:
    state_code = tweet['place']['full_name'][-2:]
    if state_code in WEST:
        west_tweets.append(tweet)
    elif state_code in MIDWEST:
        midwest_tweets.append(tweet)
    elif state_code in SOUTH:
        south_tweets.append(tweet)
    elif state_code in NORTHEAST:
        west_tweets.append(tweet)

west_tweets_text = [tweet['text'] for tweet in west_tweets]
combined_west_tweets = ' '.join(west_tweets_text)

tokenizer = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)
tokens = tokenizer.tokenize(combined_west_tweets)
tokens = [tk for tk in tokens if tk.isalnum()]
fd = nltk.FreqDist(tokens)
print(fd.most_common(20))



#http://www.nltk.org/howto/twitter.html
#https://www.nltk.org/_modules/nltk/corpus/reader/twitter.html
#https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
#https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects
#https://en.wikipedia.org/wiki/List_of_regions_of_the_United_States
#http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.casual
#https://www.nltk.org/api/nltk.twitter.html
#http://www.nltk.org/_modules/nltk/twitter/twitterclient.html