from nltk.corpus import stopwords

TWEET_COLUMNS = ["tweet_text", "profile_description", "mention_in_tweet", "retweet_count", "favorite_count",
                 "followers_count", "following_count", "gender"]

TWITTER_ROOT_URL = "https://api.twitter.com"

TWITTER_CONSUMER_KEY = "3RZxLkkQFDMnN3epDPOcP61hP"
TWITTER_CONSUMER_SECRET = "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk"

PUNCTUATION_TABLE = str.maketrans("", "", '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')
STOP_WORDS = stopwords.words("english")
