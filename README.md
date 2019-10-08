
# Tweet Sentiment Analyzer
Scrapes tweets for a particular Twitter user with Selenium and Tweepy

Analyzes sentiment and frequency of tweets using NLTK
* n-gram frequency over time
* positive, negative, and neutral sentiment

## Usage
* add [Twitter access tokens](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens) in `scraping_scripts/api_keys.json`
* change desired user and time period in `scraping_scripts/scrape.py` and `scraping_scripts/get_metadata.py`
* change sentiment thresholds in `data/gather_data.py`

