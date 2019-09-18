# import needed packages
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

# open twitter developer website and create new app
# after create new app go to detail of that app
# then go to Keys and Tokens and copy keys and tokens 
# consumer key, consumer secret, access token, access secret.
ckey="RNfBmdemeFzrsyCpVLJkWSfYE"
csecret="hFc0BShRHexODrdZuUP7oKuwIDR7mrvN4rx1WmBbgA2oT54rg9"
atoken="3273760166-mHNJIOUfAcwjjOWoh6238YZaQ9WVBxcFDXcHloe"
asecret="1BdRKzLEPar15ES4Oy6I3SX5kKkxpIp6Do7RcEftGh6Ze"

# create listener class which extends StreamListener Class
class listener(StreamListener):

	# create function for analyze Sentiment
	def on_data(self, data):
		try:
			# load data from twitter usinf json
			all_data = json.loads(data)

			# store text from data in tweet
			tweet = all_data["text"]

			# find vote and confidence using sentiment function
			sentiment_value, confidence = s.sentiment(tweet)

			# print vote and confidence
			print(tweet, sentiment_value, confidence)

			# if confidenve greater than 0.8
			if confidence*100 >= 80:
				
				# open text file in append mode
				output = open("Test/india.txt", "a")

				# write sentiment_value(vote) in file
				output.write(sentiment_value)
				output.write("\n")

				# close file
				output.close()

			return True
		except:
			return True

	# create function for show any errors if occurs
	def on_error(self, status):
		print(status)

# authenticate app keys and tokens
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

# create twitterStream with app info and listener class
twitterStream = Stream(auth, listener())

# track any word using twitterStream
twitterStream.filter(track=["india"])