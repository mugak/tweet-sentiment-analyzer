import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import dateparser
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
import string

def sort_tweets_by_sentiment(): #sorts tweets into positive, negative, and neutral using VADER
    all_tweets = {'positive_tweets':[], 'negative_tweets':[], 'neutral_tweets':[]}
    POSITIVE_THRESHOLD = 0.05 
    NEGATIVE_THRESHOLD = -0.05 
    TWEET_CORPUS_FILE = 'elonmusk_short.json'

    with open(TWEET_CORPUS_FILE, 'r') as file_name:
        tweet_corpus = json.load(file_name) #loads complete tweet corpus

    analyzer = SentimentIntensityAnalyzer()

    for tweet in tweet_corpus: #if the tweet meets the assigned threshold, label it positive, negative, or neutral
        sentiment_scores = analyzer.polarity_scores(tweet['text'])
        compound_score = sentiment_scores['compound']
        if compound_score >= POSITIVE_THRESHOLD:
            all_tweets['positive_tweets'].append(tweet)
        elif compound_score <= NEGATIVE_THRESHOLD:
            all_tweets['negative_tweets'].append(tweet)
        else:
            all_tweets['neutral_tweets'].append(tweet)

    for string_name, dict_name in all_tweets.items(): #creates a corpus for each type of tweet (pos, neg, or neutral)
        with open(string_name + '.txt', 'w') as file_name:
            json.dump(dict_name, file_name)

def sentiment_freq_over_time():
    all_tweets = load_tweets() #loads corpora for each type of tweet (pos, neg, neutral)

    sentiment_results = open('sentiment_results.txt', 'w+')
    sentiment_results.write('total number of positive tweets: {}\ntotal number of negative tweets: {}\ntotal number of neutral tweets: {}\n'
                .format(len(all_tweets['positive_tweets']), len(all_tweets['negative_tweets']), len(all_tweets['neutral_tweets']))) #writes total count of each type of tweet
    

    for tweet_type, tweet_list in all_tweets.items(): #counts num tweets of each type for each month
        month_year = {}
        for year in range(2010, 2020):
            month_year.update({(month, year):0 for month in range(1,13)}) #dictionary with a (month, year) tuple as the key and 0 (count) as the value
        for tweet in tweet_list:
            date = re.sub(r'\+\d\d\d\d ', '', tweet['created_at']) #filters out the milliseconds
            parsed_date = dateparser.parse(date)
            month_year[(parsed_date.month, parsed_date.year)] += 1 #increments number of tweets for that month

        sentiment_results.write('number of {} by month and year:\n'.format(tweet_type))
        for (month, year), count in month_year.items(): #writes num tweets of each type for each month
            sentiment_results.write('{}/{}: {}\n'.format(month, year, count))
    sentiment_results.close()

def get_unigrams_from_sentiment():
    all_tweets = load_tweets()

    english_stopwords = stopwords.words('english')
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    unigram_results = open('unigram_results.txt', 'w+')
    unigram_results.write('top 20 most frequent unigrams\n')
    for tweet_type, tweet_list in all_tweets.items(): 
        list_text = [tweet['text'] for tweet in tweet_list] #puts all tweet text into a list
        combined_text = ' '.join(list_text) #joins tweet text into one string
        tokens = tknzr.tokenize(combined_text)
        english_stopwords.extend(['…','’','...','“','”','️']) #adds punctuation not included in string.punctuation
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation] #filters out stopwords and punctuation
        fd = nltk.FreqDist(tokens)
        unigram_results.write('{}\n{}\n'.format(tweet_type, fd.most_common(20))) #writes 20 most frequent unigrams for each tweet type

    unigram_results.close()

def get_bigrams_from_sentiment():
    all_tweets = load_tweets()
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    english_stopwords = stopwords.words('english')
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    bigram_results = open('bigram_results.txt', 'w+')
    bigram_results.write('top 10 most frequent bigrams\n')
    for tweet_type, tweet_list in all_tweets.items():
        list_text = [tweet['text'] for tweet in tweet_list]
        combined_text = ' ngramwordblocker '.join(list_text) #joins tweet text, delimiting each tweet so there are no bigrams between tweets
        tokens = tknzr.tokenize(combined_text)
        english_stopwords.extend(['…','’','...','“','”','️'])
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation]
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_word_filter(lambda w: w in 'ngramwordblocker') #filters out delimiter
        bigrams = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10] #gets 10 most frequent bigrams

        bigram_results.write('{}\n{}\n'.format(tweet_type, bigrams))

    bigram_results.close()

def get_trigrams_from_sentiment():
    all_tweets = load_tweets()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    english_stopwords = stopwords.words('english')
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    trigram_results = open('trigram_results.txt', 'w+')
    trigram_results.write('top 10 most frequent trigrams\n')
    for tweet_type, tweet_list in all_tweets.items():
        list_text = [tweet['text'] for tweet in tweet_list]
        combined_text = ' ngramwordblocker '.join(list_text) #joins tweet text, delimiting each tweet so there are no trigrams between tweets
        tokens = tknzr.tokenize(combined_text)
        english_stopwords.extend(['…','’','...','“','”','️'])
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation]
        finder = TrigramCollocationFinder.from_words(tokens)
        finder.apply_word_filter(lambda w: w in 'ngramwordblocker') #filters out delimiter
        trigrams = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10] #gets 10 most frequent trigrams

        trigram_results.write('{}\n{}\n'.format(tweet_type, trigrams)) 

    trigram_results.close()
       
def load_tweets(): #loads corpus for each type of tweet (pos, neg, neutral)
    all_tweets = {'positive_tweets':[], 'negative_tweets':[], 'neutral_tweets':[]}
    for tweets_file in all_tweets.keys():
        with open(tweets_file + '.txt', 'r') as file_name:
            all_tweets[tweets_file] = json.load(file_name)
    return all_tweets

def get_unigrams_over_time():
    TWEET_CORPUS_FILE = 'elonmusk_short.json'
    with open(TWEET_CORPUS_FILE, 'r') as file_name:
        tweet_corpus = json.load(file_name)

    english_stopwords = stopwords.words('english')
    english_stopwords.extend(['…','’','...','“','”','️'])
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    unigram_results = open('unigrams_over_time.txt', 'w+')
    unigram_results.write('top 20 most frequent unigrams by year\n')

    years = {year:[] for year in range(2010, 2020)} #dictionary with the year as key, list of tweets as value
    for tweet in tweet_corpus:
        date = re.sub(r'\+\d\d\d\d ', '', tweet['created_at']) #filters out milliseconds
        parsed_date = dateparser.parse(date)
        years[parsed_date.year].append(tweet) #puts tweet into the matching year bucket

    for year, tweet_list in years.items(): #combines tweets for each year, get most frequent unigrams for that year
        list_text = [tweet['text'] for tweet in tweet_list]
        combined_text = ' '.join(list_text)
        tokens = tknzr.tokenize(combined_text)
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation]
        fd = nltk.FreqDist(tokens)
        unigram_results.write('{}\n{}\n'.format(year,fd.most_common(20)))

    unigram_results.close()

def get_bigrams_over_time():
    TWEET_CORPUS_FILE = 'elonmusk_short.json'
    with open(TWEET_CORPUS_FILE, 'r') as file_name:
        tweet_corpus = json.load(file_name)

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    english_stopwords = stopwords.words('english')
    english_stopwords.extend(['…','’','...','“','”','️'])
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    bigram_results = open('bigrams_over_time.txt', 'w+')
    bigram_results.write('top 10 most frequent bigrams by year\n')

    years = {year:[] for year in range(2010, 2020)} #dictionary with the year as key, list of tweets as value
    for tweet in tweet_corpus:
        date = re.sub(r'\+\d\d\d\d ', '', tweet['created_at']) #filters out milliseconds
        parsed_date = dateparser.parse(date)
        years[parsed_date.year].append(tweet) #puts tweet into the matching year bucket

    for year, tweet_list in years.items(): #combines tweets for each year, get most frequent bigrams for that year
        list_text = [tweet['text'] for tweet in tweet_list]
        combined_text = ' ngramwordblocker '.join(list_text)
        tokens = tknzr.tokenize(combined_text)
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation]
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_word_filter(lambda w: w in 'ngramwordblocker')
        bigrams = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10] 

        bigram_results.write('{}\n{}\n'.format(year, bigrams))

    bigram_results.close()

def get_trigrams_over_time():
    TWEET_CORPUS_FILE = 'elonmusk_short.json'
    with open(TWEET_CORPUS_FILE, 'r') as file_name:
        tweet_corpus = json.load(file_name)

    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    english_stopwords = stopwords.words('english')
    english_stopwords.extend(['…','’','...','“','”','️'])
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    trigram_results = open('trigrams_over_time.txt', 'w+')
    trigram_results.write('top 10 most frequent trigrams by year\n')

    years = {year:[] for year in range(2010, 2020)} #dictionary with the year as key, list of tweet as value
    for tweet in tweet_corpus:
        date = re.sub(r'\+\d\d\d\d ', '', tweet['created_at']) #filters out milliseconds
        parsed_date = dateparser.parse(date)
        years[parsed_date.year].append(tweet) #puts tweet into the matching year bucket

    for year, tweet_list in years.items(): #combines tweets for each year, get most frequent trigrams for that year
        list_text = [tweet['text'] for tweet in tweet_list]
        combined_text = ' ngramwordblocker '.join(list_text)
        tokens = tknzr.tokenize(combined_text)
        tokens = [tk for tk in tokens if tk not in english_stopwords and tk not in string.punctuation]
        finder = TrigramCollocationFinder.from_words(tokens)
        finder.apply_word_filter(lambda w: w in 'ngramwordblocker')
        trigrams = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10] 

        trigram_results.write('{}\n{}\n'.format(year, trigrams))

    trigram_results.close()

### SOURCES ###
#https://github.com/bpb27/twitter_scraping
#https://www.nltk.org/_modules/nltk/sentiment/vader.html
#https://github.com/cjhutto/vaderSentiment
#https://developers.google.com/edu/python/regular-expressions
#https://dateparser.readthedocs.io/en/latest/
#https://docs.python.org/3/library/datetime.html
#https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
#http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.casual
#https://www.nltk.org/book/ch02.html#code-unusual
#http://www.nltk.org/howto/collocations.html
#https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
