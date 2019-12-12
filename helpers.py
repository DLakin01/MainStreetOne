import matplotlib.pyplot as plt
import numpy as np
import spacy
import json
import time
import re

from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS
from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import learning_curve
from gensim.corpora import Dictionary
from tarfile import ExFileObject
from textblob import TextBlob

from constants import CONTRACTIONS, POS_MAP, STOP_WORDS


def clean_text(series):
    nlp = spacy.load("en", disable=["tagger", "ner", "parser"])
    parsed_sentences = []
    for doc in nlp.tokenizer.pipe(series, batch_size=10000):
        parsed_text = []
        for token in doc:
            if token.text in ["http", "https", "rt", "RT", "tco", "co"]:
                continue
            elif token.text.startswith("@"):
                continue
            elif token.text.startswith("#"):
                continue
            elif token.text in STOP_WORDS:
                continue

            token.text.replace("\n", "")
            parsed_text.append(token.text.lower())
        parsed_sentences.append(parsed_text)

    return parsed_sentences


def bigrams(words, bi_min=15):
    bigram = Phrases(words, min_count=bi_min)
    bigram_mod = Phraser(bigram)
    return bigram_mod


def get_corpus(df):
    words = clean_text(df["combined_text"].values)
    bigram = bigrams(words)
    bigram = [bigram[tweet] for tweet in words]
    id2word = Dictionary(bigram)
    id2word.filter_extremes(no_below=50, no_above=0.40)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        kw['log_time'].append(int((te - ts) * 1000))
        return result

    return timed


def parse_julia_file(tarfile: ExFileObject):
    tar_string = tarfile.read().decode()
    return [json.loads(el) for el in tar_string.split("\n") if el != ""]


def parse_tweet_text(tweet_obj: dict):
    try:
        text = tweet_obj["full_text"]
        if text == "":
            text = tweet_obj["text"]

    except KeyError:
        text = tweet_obj["text"]

    return text


def extract_linguistic_features(texts, tweet_ids, spacy_nlp):
    all_features = []

    for i, doc in enumerate(spacy_nlp.pipe(texts, n_threads=16, batch_size=10000)):
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


def plot_learning_curve(estimator, title, X, y):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 5), return_times=True
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    axes[0].fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )

    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(
        train_sizes, fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std, alpha=0.1
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(
        fit_times_mean, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

