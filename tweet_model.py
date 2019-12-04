import pandas as pd
import numpy as np
import tarfile

from random import shuffle

from text_helpers import parse_julia_file, parse_tweet_text

TWEET_COLUMNS = ["user_id", "tweet_text", "hashtags", "users_mentioned_name", "users_mentioned_screen_name",
                 "user_replied_to", "rt_text", "quote_text", "gender"]

TWITTER_CONSUMER_KEY = "3RZxLkkQFDMnN3epDPOcP61hP"
TWITTER_CONSUMER_SECRET = "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk"

TWITTER_ACCESS_TOKEN = "1972482236-r6kxpQXXcBKTIRdwPXGjEzhLN2v5asQaO6s4kA3"
TWITTER_ACCESS_SECRET = "RkW6alMhS7AIIE70rTfDtlvpiCZOl7bDAw5iBqqOob2HU"

tar = tarfile.open("mso_ds_interview.tgz", "r")
manifest = tar.extractfile("ds_interview/manifest.jl")
manifest_details = parse_julia_file(manifest)

# Special shuffling and segmentation for ease of testing
# shuffle(manifest_details)
# manifest_details = manifest_details[:10]

tweets_data_list = []

for user in manifest_details:
    print(f"Pulling tweets from user {user['user_id_str']}")
    tweet_file = tar.extractfile(f"ds_interview/tweet_files/{user['user_id_str']}.tweets.jl")
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
            user["user_id_str"],
            tweet_text,
            hashtags,
            users_mentioned,
            users_mentioned_sn,
            user_replied_to,
            rt_text,
            quote_text,
            user["gender_human"]
        ])

shuffle(tweets_data_list)

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)
tweets_df.to_csv("C:\\Users\\DLaki\\OneDrive\\Desktop\\Github\\ms1_df.csv", index=False)

# Apply transforms to dataframe
gender_map = {"M": 0, "F": 1}
tweets_df["gender"] = tweets_df["gender"].map(gender_map)

tar.close()
