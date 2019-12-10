import pandas as pd
import numpy as np
import requests
import tarfile
import spacy

from requests.auth import HTTPBasicAuth
from spacymoji import Emoji

from constants import TWEET_COLUMNS, TWITTER_ROOT_URL, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET
from helpers import parse_julia_file, parse_tweet_text, extract_features

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

        else:
            profile_description = api_result[0]["description"]

        tweet_file = tar.extractfile(f"ds_interview/tweet_files/{user_id}.tweets.jl")
        user_tweets = parse_julia_file(tweet_file)

        for t in user_tweets:
            tdoc = t["document"]
            combined_text = parse_tweet_text(tdoc) + " " + profile_description
            has_mentions = int(bool(tdoc["entities"]["user_mentions"]))
            num_mentions = len(tdoc["entities"]["user_mentions"])
            has_hashtags = int(bool(tdoc["entities"]["hashtags"]))
            num_hashtags = len(tdoc["entities"]["hashtags"])

            tweets_data_list.append([
                combined_text,
                has_mentions,
                num_mentions,
                has_hashtags,
                num_hashtags,
                user["gender_human"]
            ])

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)

# Take a sample of the dataset for ease of processing
tweets_df = tweets_df.sample(frac=0.1, random_state=1)

# Cleaning and feature extraction
nlp = spacy.load("en")
emoji = Emoji(nlp, merge_spans=False)
nlp.add_pipe(emoji, first=True)

tweets_df.join(pd.DataFrame.from_records(
    tweets_df.apply(
        lambda x: extract_features(x["tweet_and_profile_text"], x["language"], nlp), axis=1
    ).values, index=tweets_df.index)
)



# Pull out labels
gender_map = {"M": 0, "F": 1}
tweets_df["gender_binary"] = tweets_df["gender"].map(gender_map)
