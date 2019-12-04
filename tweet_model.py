import pandas as pd
import numpy as np
import tarfile

from text_helpers import parse_julia_file, parse_tweet_text

TWEET_COLUMNS = ["user_id", "tweet_text", "hashtags", "users_mentioned_name", "users_mentioned_screen_name",
                 "user_replied_to", "rt_text", "quote_text"]

tar = tarfile.open("mso_ds_interview.tgz", "r")
manifest = tar.extractfile("ds_interview/manifest.jl")
manifest_details = parse_julia_file(manifest)

user_ids = [el["user_id_str"] for el in manifest_details]

tweets_data_list = []

for user in user_ids:
    print(f"Pulling tweets from user {user}")
    tweet_file = tar.extractfile(f"ds_interview/tweet_files/{user}.tweets.jl")
    user_tweets = parse_julia_file(tweet_file)

    for t in user_tweets:
        tdoc = t["document"]

        tweet_text = parse_tweet_text(tdoc)
        hashtags = " | ".join([h["text"] for h in tdoc["entities"]["hashtags"]])
        users_mentioned = " | ".join([u["name"] for u in tdoc["entities"]["user_mentions"]])
        users_mentioned_sn = " | ".join(u["screen_name"] for u in tdoc["entities"]["user_mentions"])
        user_replied_to = tdoc["in_reply_to_screen_name"]

        rt_text = ""
        quote_text = ""

        if "retweeted_status" in tdoc:
            if tdoc["retweeted_status"] is not None:
                rt_text = parse_tweet_text(tdoc["retweeted_status"])

                if "quoted_status" in tdoc["retweeted_status"]:
                    if tdoc["retweeted_status"]["quoted_status"] is not None:
                        quote_text = parse_tweet_text(tdoc["retweeted_status"]["quoted_status"])

        tweets_data_list.append([
            user,
            tweet_text,
            hashtags,
            users_mentioned,
            users_mentioned_sn,
            user_replied_to,
            rt_text,
            quote_text
        ])

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)
print("ok")
