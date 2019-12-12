import pandas as pd
import numpy as np
import string
import json
import time
import re

from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS
from tarfile import ExFileObject
from textblob import TextBlob

from constants import CONTRACTIONS, POS_MAP


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
