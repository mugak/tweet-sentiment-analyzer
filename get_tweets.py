import json, nltk
from nltk.corpus import twitter_samples

#place = json.loads(..)
#full_name = place[full_name]
#state_code = full_name[-2:]
geotagged_tweets, west_tweets, midwest_tweets, south_tweets, northeast_tweets = [], [], [], [], []
WEST = ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'AK', 'HI']
MIDWEST = ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'MI', 'IN', 'OH']
SOUTH = ['TX', 'OK', 'AR', 'LA', 'KY', 'TN', 'MS', 'AL', 'WV', 'WD', 'DE', 'DC', 'VA', 'NC', 'SC', 'GA', 'FL']
NORTHEAST = ['NY', 'PA', 'NJ', 'ME', 'VT', 'NH', 'MA', 'CT', 'RI']

for tweet in twitter_samples.docs():
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
print(len(west_tweets))

#http://www.nltk.org/howto/twitter.html
#https://www.nltk.org/_modules/nltk/corpus/reader/twitter.html
#https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
#https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects
#https://en.wikipedia.org/wiki/List_of_regions_of_the_United_States