import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import tarfile
import pickle

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_predict
from gensim.models.ldamulticore import LdaMulticore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from requests.auth import HTTPBasicAuth

from constants import TWEET_COLUMNS, TWITTER_ROOT_URL, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, FILE_PATH
from helpers import parse_julia_file, parse_tweet_text, get_corpus, plot_learning_curve


if __name__ == "__main__":
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
                tweet_id = tdoc["id"]
                combined_text = parse_tweet_text(tdoc) + " " + profile_description
                language = tdoc["lang"]
                has_mentions = int(bool(tdoc["entities"]["user_mentions"]))
                num_mentions = len(tdoc["entities"]["user_mentions"])
                has_hashtags = int(bool(tdoc["entities"]["hashtags"]))
                num_hashtags = len(tdoc["entities"]["hashtags"])

                tweets_data_list.append([
                    tweet_id,
                    combined_text,
                    language,
                    has_mentions,
                    num_mentions,
                    has_hashtags,
                    num_hashtags,
                    user["gender_human"]
                ])

    tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)
    tweets_df.set_index("id", inplace=True)
    tweets_df.to_csv(f"{FILE_PATH}\\ms1_df.csv")

    # tweets_df = pd.read_csv(f"{FILE_PATH}\\ms1_df.csv", dtype={"combined_text": str})
    # tweets_df = tweets_df.sample(frac=0.1, random_state=1)
    #
    # english_only_df = tweets_df[tweets_df["language"] == "en"]
    #
    # train_corpus, train_id2word, train_bigram = get_corpus(english_only_df)
    # lda_train = LdaMulticore(
    #     corpus=train_corpus,
    #     num_topics=20,
    #     id2word=train_id2word,
    #     passes=3,
    #     workers=2
    # )
    #
    # train_vecs = []
    # for i in range(len(english_only_df.index)):
    #     top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
    #     topic_vec = [top_topics[j][1] for j in range(20)]
    #     topic_vec.append(english_only_df.iloc[i]["num_mentions"])
    #     topic_vec.append(english_only_df.iloc[i]["num_hashtags"])
    #     topic_vec.append(len(english_only_df.iloc[i]["combined_text"]))
    #     train_vecs.append(topic_vec)
    #
    # X = np.array(train_vecs)
    # y = np.array(english_only_df["gender"].map({"M": 0, "F": 1}))

    with open("y.pkl", "rb") as pkl:
        y = pickle.load(pkl)

    with open("X.pkl", "rb") as pkl:
        X = pickle.load(pkl)

    lr = LogisticRegression(
        penalty="l2",
        random_state=1,
        max_iter=1000
    )
    plot_learning_curve(lr, "Learning Curve with Logistic Regression", X, y)

    # Stochastic Gradient Descent
    sgd = SGDClassifier(
        early_stopping=True,
        random_state=1
    )
    plot_learning_curve(sgd, "Learning Curve with Stochastic Gradient Descent", X, y)

    plt.show()

    print("ok")
