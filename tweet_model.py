import pandas as pd
import numpy as np
import requests
import tarfile
import spacy

from requests.auth import HTTPBasicAuth
from spacymoji import Emoji

from constants import TWEET_COLUMNS, TWITTER_ROOT_URL, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, FILE_PATH, \
    SPACY_LANGS
from helpers import parse_julia_file, parse_tweet_text, extract_linguistic_features

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
            language = tdoc["lang"]
            has_mentions = int(bool(tdoc["entities"]["user_mentions"]))
            num_mentions = len(tdoc["entities"]["user_mentions"])
            has_hashtags = int(bool(tdoc["entities"]["hashtags"]))
            num_hashtags = len(tdoc["entities"]["hashtags"])

            tweets_data_list.append([
                combined_text,
                language,
                has_mentions,
                num_mentions,
                has_hashtags,
                num_hashtags,
                user["gender_human"]
            ])

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)

# Cleaning and feature extraction
# Throw out samples that spaCy can't parse
tweets_df = tweets_df[tweets_df["language"].isin(SPACY_LANGS)]

split_by_lang = [{"lang": lang, "df": tweets_df[tweets_df["language"] == lang]} for lang in SPACY_LANGS]

for item in split_by_lang:
    nlp = spacy.load(item["lang"])
    emoji = Emoji(nlp, merge_spans=False)
    nlp.add_pipe(emoji, first=True)

    texts = item["df"]["combined_text"].tolist()
    spacy_features = extract_linguistic_features(texts, nlp)

    temp_df = pd.DataFrame.from_records(spacy_features)

    item["df"] = item["df"].join(pd.DataFrame.from_records(spacy_features))

tweets_df = pd.concat([lang_item["df"] for lang_item in split_by_lang])
tweets_df.to_csv(f"{FILE_PATH}\\ms1_df.csv")

