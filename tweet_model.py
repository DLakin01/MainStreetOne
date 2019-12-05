import pandas as pd
import numpy as np
import requests
import tarfile

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from requests.auth import HTTPBasicAuth
from random import shuffle

from helpers import parse_julia_file, parse_tweet_text
from constants import TWEET_COLUMNS, TWITTER_ROOT_URL, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET

tar = tarfile.open("mso_ds_interview.tgz", "r")
manifest = tar.extractfile("ds_interview/manifest.jl")
manifest_details = parse_julia_file(manifest)


tweets_data_list = []

# Authenticate to Twitter API
auth = HTTPBasicAuth(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth_response = requests.post(f"{TWITTER_ROOT_URL}/oauth2/token?grant_type=client_credentials", auth=auth).json()
headers = {"Authorization": f"Bearer {auth_response['access_token']}"}

with requests.session() as session:
    session.headers = headers

    for user in manifest_details:
        user_id = user["user_id_str"]
        print(f"Pulling tweets from user {user_id}")

        api_result = session.get(f"{TWITTER_ROOT_URL}/1.1/users/lookup.json?user_id={user_id}").json()
        if "errors" in api_result:
            profile_description = ""
            user_followers_count = ""
            user_following_count = ""

        else:
            user_info = api_result[0]

            profile_description = user_info["description"]
            user_followers_count = user_info["followers_count"]
            user_following_count = user_info["friends_count"]

        tweet_file = tar.extractfile(f"ds_interview/tweet_files/{user_id}.tweets.jl")
        user_tweets = parse_julia_file(tweet_file)

        for t in user_tweets:
            tdoc = t["document"]
            tweet_text = parse_tweet_text(tdoc)
            retweet_count = tdoc["retweet_count"]
            favorite_count = tdoc["favorite_count"]
            user_mentions = int(bool(tdoc["entities"]["user_mentions"]))

            tweets_data_list.append([
                tweet_text,
                profile_description,
                user_mentions,
                retweet_count,
                favorite_count,
                user_followers_count,
                user_following_count,
                user["gender_human"]
            ])

shuffle(tweets_data_list)

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)

# Transform text data to TF-IDF


# Pull out labels
gender_map = {"M": 0, "F": 1}
y = tweets_df["gender"].map(gender_map)
tweets_df.drop("gender", axis=1, inplace=True)

# Separate training, validation, and test data
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(tweets_df, y)

tar.close()
