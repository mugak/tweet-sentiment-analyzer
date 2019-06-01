import nltk.twitter

oauth = nltk.twitter.credsfromfile()
client = nltk.twitter.Streamer(**oauth)
client.register(nltk.twitter.TweetWriter(limit=10))
client.statuses.sample()