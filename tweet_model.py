# <markdowncell>
# # Tweet Gender Classification with NLP, Scikit-Learn, and LightGBM
# ### *Daniel Lakin*
# ---
# In this notebook, we use various a combination of machine learning and linguistic processing tools to build
# a model that can predict, given an individual tweet, the author's gender. Working from an archived set of files,
# we extract raw data on tweets from 200 individual users, and supplement the data with a call to Twitter's API. We
# then extract new linguistic and categorical features on each tweet, working primarily from the posted text.
#
# After feature extraction and scaling, we feed it into a series of Microsoft LightGBM Gradient Boosting Classifiers,
# taking advantage of the power and speed offered by the algorithm. The models are first tuned using a randomized search
# cross-validation approach, and then refined by conducting a grid search around the results of the random search.
#
# To run the code in this notebook, please ensure you have the following Python packages installed, either via pip or in
# a conda environment:
#
# - matplotlib
# - seaborn
# - lightgbm
# - sklearn
# - requests
# - pandas
# - numpy
# - spacy
# - spacymoji
# - textblob

# <codecell>
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import tarfile
import random
import spacy
import json
import re

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from requests.auth import HTTPBasicAuth
from sklearn.pipeline import Pipeline
from tarfile import ExFileObject
from textblob import TextBlob
from itertools import product
from spacymoji import Emoji
from pprint import pprint

TWEET_COLUMNS = ["id", "combined_text", "language", "num_mentions", "num_hashtags", "gender"]
TWITTER_ROOT_URL = "https://api.twitter.com"
TWITTER_CONSUMER_KEY = "3RZxLkkQFDMnN3epDPOcP61hP"
TWITTER_CONSUMER_SECRET = "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk"


# <markdowncell>
# In the following section, we unzip the archive containing our data, and parse the `manifest.jl` file, which contains
# unique IDs for each user in the dataset, as well as their gender, denoted by M for male and F for female. We'll be
# using the gender data as our label data later in the process.

# <codecell>
def parse_julia_file(tarfile: ExFileObject):
    tar_string = tarfile.read().decode()
    return [json.loads(el) for el in tar_string.split("\n") if el != ""]


tar = tarfile.open("mso_ds_interview.tgz", "r")
manifest = tar.extractfile("ds_interview/manifest.jl")
manifest_details = parse_julia_file(manifest)

# <markdowncell>
# Using the unique IDs extracted from manifest.jl, we can then use the same parse_julia_file method defined above to
# pull the information for each of the 200 users. For each user, we also pull their profile description via the
# Twitter API. Once we have processed all of the 700,000+ tweets in this dataset, we assemble a Pandas dataframe,
# containing the following information for each tweet:
#
# - Unique tweet ID
# - Text of tweet + the user's profile description
# - The language the tweet was written in
# - The number of mentions in the tweet
# - The number of hashtags in the tweet

# <codecell>
tweets_data_list = []

# Authenticate to Twitter API
auth = HTTPBasicAuth("3RZxLkkQFDMnN3epDPOcP61hP", "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk")
auth_response = requests.post("https://api.twitter.com/oauth2/token?grant_type=client_credentials", auth=auth).json()
headers = {"Authorization": f"Bearer {auth_response['access_token']}"}


def parse_tweet_text(tweet_obj: dict):
    try:
        text = tweet_obj["full_text"]
        if text == "":
            text = tweet_obj["text"]

    except KeyError:
        text = tweet_obj["text"]

    return text


with requests.session() as session:
    session.headers = headers

    for user in manifest_details:
        user_id = user["user_id_str"]

        api_result = session.get(f"https://api.twitter.com/1.1/users/lookup.json?user_id={user_id}").json()
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
            num_mentions = len(tdoc["entities"]["user_mentions"])
            num_hashtags = len(tdoc["entities"]["hashtags"])

            tweets_data_list.append([
                tweet_id,
                combined_text,
                language,
                num_mentions,
                num_hashtags,
                user["gender_human"]
            ])

tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)
tweets_df.set_index("id", inplace=True)
tweets_df.head()

# <markdowncell>
# Before proceeding, let's check to see the distribution of languages across our dataset.

# <codecell>
language_counts = tweets_df.groupby("language").count()["combine_text"]
print(language_counts)

# <markdowncell>
# As we can see, there are over a dozen languages present in the dataset, with English being the vast majority (over
# 88%). While this introduces some complexity, we can cope with it - the excellent NLP library spaCy provides strong
# support across multiple languages. In the interest of preserving as much of the data as possible, we'll redefine our
# dataframe to include tweets only in the languages spaCy can "understand", and proceed with linguistic feature
# extraction.

# <codecell>
SPACY_LANGS = ["de", "el", "en", "es", "fr", "it", "lt", "nb", "nl", "pt"]

# Throw out samples that spaCy can't parse
tweets_df = tweets_df[tweets_df["language"].isin(SPACY_LANGS)]

# <markdowncell>
# After narrowing the dataset based on language, we now take another step to reduce the size of the data we're working
# with. Even with only spaCy-friendly languages, the dataset still contains over 600,000 tweets, which will be very
# computationally expensive to work with. To make things easier but still keep the modeling process robust, we'll take
# a 20% sample of the data and use that from now on.

# <codecell>
tweets_df = tweets_df.sample(frac=0.2, random_state=42)

# <markdowncell>
# With the size of the data reduced to a more manageable level, we can begin our linguistic processing and feature
# extraction. In the below section, we work with one language group at a time, running each one through a suite of
# linguistic checks, leveraging spaCy's ability to parse out parts of speech and other features from our chosen
# languages. We also collect information on the number of contractions and emoticons in each tweet, as well as it's
# sentiment as detected by the Textblob library's parser.

# <codecell>
CONTRACTIONS = ["ain't", "aren't", "can't", "can't've", "'cause", "could've", "couldn't", "couldn't've", "didn't",
                "doesn't", "don't", "hadn't", "hadn't've", "hasn't", "haven't", "he'd", "he'd've", "he'll", "he'll've",
                "he's", "how'd", "how'd'y", "how'll", "how's", "I'd", "I'd've", "I'll", "I'll've", "I'm", "I've", "i'd",
                "i'd've", "i'll", "i'll've", "i'm", "i've", "isn't", "it'd", "it'd've", "it'll", "it'll've", "it's",
                "let's", "ma'am", "mayn't", "might've", "mightn't", "mightn't've", "must've", "mustn't", "mustn't've",
                "needn't", "needn't've", "o'clock", "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've",
                "she'd", "she'd've", "she'll", "she'll've", "she's", "should've", "shouldn't", "shouldn't've", "so've",
                "so's", "that'd", "that'd've", "that's", "there'd", "there'd've", "there's", "they'd", "they'd've",
                "they'll", "they'll've", "they're", "they've", "to've", "wasn't", "we'd", "we'd've", "we'll",
                "we'll've", "we're", "we've", "weren't", "what'll", "what'll've", "what're", "what's", "what've",
                "when's", "when've", "where'd", "where's", "where've", "who'll", "who'll've", "who's", "who've",
                "why's", "why've", "will've", "won't", "won't've", "would've", "wouldn't", "wouldn't've", "y'all",
                "y'all'd", "y'all'd've", "y'all're", "y'all've", "you'd", "you'd've", "you'll", "you'll've", "you're",
                "you've"]
POS_MAP = {
    "ADJ": "num_adjectives",
    "ADV": "num_adverbs",
    "CONJ": "num_conjunctions",
    "NOUN": "num_nouns",
    "NUM": "num_numerals",
    "PART": "num_particles",
    "PRON": "num_pronouns",
    "PROPN": "num_proper_nouns",
    "PUNCT": "num_punctuation_mks",
    "VERB": "num_verbs"
}


def extract_linguistic_features(texts, tweet_ids, spacy_nlp):
    all_features = []

    for i, doc in enumerate(spacy_nlp.pipe(texts, disable=["tagger", "parser", "ner"], n_threads=16, batch_size=10000)):
        features = {
            "id": tweet_ids[i],
            "num_words": len(doc),
            "tweet_length": len(doc.text),
            "num_exclamation_pts": doc.text.count("!"),
            "num_question_mks": doc.text.count("?"),
            "num_periods": doc.text.count("."),
            "num_hyphens": doc.text.count("-"),
            "num_capitals": sum(1 for char in doc.text if char.isupper()),
            "num_emoticons": sum(
                1 for token in doc
                if token._.is_emoji
                or token in UNICODE_EMOJI
                or token in UNICODE_EMOJI_ALIAS
            ),
            "num_unique_words": len(set(token.text for token in doc)),
            "num_adjectives": 0,
            "num_nouns": 0,
            "num_pronouns": 0,
            "num_adverbs": 0,
            "num_conjunctions": 0,
            "num_numerals": 0,
            "num_particles": 0,
            "num_proper_nouns": 0,
            "num_verbs": 0,
            "num_contractions": 0,
            "num_punctuation_mks": 0
        }

        for token in doc:
            if token.text in CONTRACTIONS:
                features["num_contractions"] += 1

            if token.pos_ in POS_MAP:
                column_key = POS_MAP[token.pos_]
                features[column_key] += 1

        clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", doc.text).split())
        features["sentiment"] = parse_sentiment(clean_tweet)

        all_features.append(features)

    return all_features


def parse_sentiment(tweet):
    """
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    """

    # Create TextBlob object of tweet text
    parsed = TextBlob(tweet)
    # set sentiment
    if parsed.sentiment.polarity > 0:
        return 1
    elif parsed.sentiment.polarity == 0:
        return 0
    else:
        return -1


split_by_lang = [{"lang": lang, "df": tweets_df[tweets_df["language"] == lang]} for lang in SPACY_LANGS]
for item in split_by_lang:
    nlp = spacy.load(item["lang"])
    emoji = Emoji(nlp, merge_spans=False)
    nlp.add_pipe(emoji, first=True)

    tweet_ids = item["df"].index.tolist()
    texts = item["df"]["combined_text"].tolist()
    if len(texts):
        spacy_features = extract_linguistic_features(texts, tweet_ids, nlp)
        temp_df = pd.DataFrame.from_records(spacy_features)
        temp_df.set_index("id", inplace=True)

        item["df"] = item["df"].merge(temp_df, how="left", on="id")

# <markdowncell>
# After all linguistic features have been extracted, we combine the various language-specific dataframes back into
# one, and begin scaling. We've collected a wide range of numeric data, and have categorical data in the form of the
# Textblob sentiment scores, and all need to be scaled. To do this efficiently, we set up a nested series of sklearn
# Pipelines, contained inside a ColumnTransformer, which we then use to scale all our data at once. Our numeric data
# is scaled using sklearn's RobustScaler, which helps avoid distortion due to outliers. The categorical sentiment data
# is transformed using sklearn's OneHotEncoder.

# <codecell>
X = pd.concat([lang_item["df"] for lang_item in split_by_lang])
y = np.array(X["gender"].map({"M": 0, "F": 1}))

X.drop(columns=["gender", "language", "combined_text"], inplace=True)
predictors = pd.get_dummies(X, columns=["sentiment"]).columns.tolist()

# Scale numeric features
numeric_features = ["num_mentions", "num_hashtags", "num_nouns", "num_pronouns", "num_adjectives", "num_particles",
                    "num_words", "tweet_length", "num_exclamation_pts", "num_question_mks", "num_periods", "num_verbs",
                    "num_hyphens", "num_capitals", "num_emoticons", "num_unique_words", "num_conjunctions",
                    "num_numerals", "num_contractions", "num_adverbs", "num_proper_nouns", "num_punctuation_mks"]
numeric_transformer = Pipeline(steps=[
    ("robust", RobustScaler())
])

categorical_features = ["sentiment"]
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(categories="auto"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
X = preprocessor.fit_transform(X)


def eval_lgb_results(hyperparams, iteration):
    """
    Scoring helper for grid and random search. Returns CV score from the given
    hyperparameters
    """

    # Find optimal n_estimators using early stopping
    if "n_estimators" in hyperparams.keys():
        del hyperparams["n_estimators"]

    # Perform n_folds CV
    cv_res = lgb.cv(
        params=hyperparams,
        train_set=train_set,
        num_boost_round=500,
        nfold=4,
        early_stopping_rounds=25,
        metrics="auc",
        seed=42,
        verbose_eval=True
    )

    # Return CV results
    score = cv_res["auc-mean"][-1]
    estimators = len(cv_res["auc-mean"])
    hyperparams["n_estimators"] = estimators

    return [score, hyperparams, iteration]


def light_random_search(param_grid, max_evals=5):
    # Dataframe to store results
    results = pd.DataFrame(columns=["score", "params", "iteration"], index=list(range(max_evals)))

    # Select max_eval combinations of params to check
    for i in range(max_evals):
        iter_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        results.loc[i, :] = eval_lgb_results(iter_params, i)

    # Sort by best score
    results.sort_values("score", ascending=False, inplace=True)
    results.reset_index(inplace=True)
    return results


def light_grid_search(param_grid):
    results = pd.DataFrame(columns=["score", "params", "iteration"])

    # Get every possible combination of params from the grid
    keys, grid_vals = zip(*param_grid.items())

    # Iterate over every possible combination of hyperparameters
    for i, v in enumerate(product(*grid_vals)):
        iter_params = dict(zip(keys, v))
        results.loc[i, :] = eval_lgb_results(iter_params, i)

    # Sort by best score
    results.sort_values("score", ascending=False, inplace=True)
    results.reset_index(inplace=True)
    return results


# Split into train and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20000, random_state=42)

random.seed(42)

# Convert to LGB Datasets for speed
train_set = lgb.Dataset(data=X_train, label=y_train)
test_set = lgb.Dataset(data=X_test, label=y_test)

model = lgb.LGBMModel()
default_params = model.get_params()

param_grid = {
    'boosting_type': ['gbdt'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [False]
}

random_results = light_random_search(param_grid, max_evals=10)
print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

pprint(random_results.loc[0, 'params'])

# Get the best params from the random search
rsearch_params = random_results.loc[0, "params"]

# Create, train, and test model with the derived params
model = lgb.LGBMClassifier(**rsearch_params, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("The best model from random search scores {:.5f} ROC-AUC on the test set".format(roc_auc_score(y_test, preds)))

# Plot feature importances
importances = model.feature_importances_
names = model.booster_.feature_name()

plt.figure()
plt.title("Random Search Feature Importances")
plt.bar(range(X_test.shape[1]), importances)
plt.xticks(list(range(X_test.shape[1])), labels=names, rotation=90)
plt.show()

# Use grid search to refine hyperparameters, cutting down on some
rand_leaves = rsearch_params["num_leaves"]
grid_num_leaves = range((rand_leaves - 30), (rand_leaves + 31), 20)

rand_learn_rate = rsearch_params["learning_rate"]
grid_learning_rate = np.logspace(
    np.log10(np.power(rand_learn_rate, 1.5)),
    np.log10(np.power(rand_learn_rate, .55)),
    base=10, num=4
)

rand_subsample = rsearch_params["subsample_for_bin"]
grid_subsample = range((rand_subsample - 2000), (rand_subsample + 2001), 2000)

rand_min_child = rsearch_params["min_child_samples"]
grid_min_child = range((rand_min_child - 20), (rand_min_child + 21), 15)

param_grid2 = {
    'boosting_type': ['gbdt'],
    'num_leaves': list(grid_num_leaves),
    'learning_rate': list(grid_learning_rate),
    'subsample_for_bin': list(grid_subsample),
    'min_child_samples': list(grid_min_child),
    'reg_alpha': [rsearch_params["reg_alpha"]],
    'reg_lambda': [rsearch_params["reg_lambda"]],
    'colsample_bytree': [rsearch_params["colsample_bytree"]],
    'subsample': [rsearch_params["subsample"]],
    'is_unbalance': [False]
}

grid_results = light_grid_search(param_grid2)

# Get the best params from the random search
gsearch_params = grid_results.loc[0, "params"]

# Create, train, and test model with the derived params
model = lgb.LGBMClassifier(**gsearch_params, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("The best model from grid search scores {:.5f} ROC-AUC on the test set".format(roc_auc_score(y_test, preds)))

# Plot feature importances
importances = model.feature_importances_
names = model.booster_.feature_name()

plt.figure()
plt.title("Random Search Feature Importances")
plt.bar(range(X_test.shape[1]), importances)
plt.xticks(list(range(X_test.shape[1])), labels=names, rotation=90)
plt.show()

random_results["search"] = "random"
grid_results["search"] = "grid"

all_hyper_params = random_results.append(grid_results)

best_random_hyperparams = random_results.iloc[random_results["score"].astype(np.float).idxmax()].copy()
best_grid_hyperparams = grid_results.iloc[grid_results["score"].astype(np.float).idxmax()].copy()

sns.lmplot('iteration', 'score', hue="search", data=all_hyper_params, size=8)
plt.scatter(best_random_hyperparams["iteration"], best_random_hyperparams["score"],
            marker="*", s=400, c="blue", edgecolor="k")
plt.scatter(best_grid_hyperparams["iteration"], best_grid_hyperparams["score"],
            marker="*", s=400, c="orange", edgecolor="k")
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.title("Validation ROC AUC versus Iteration")
plt.show()
