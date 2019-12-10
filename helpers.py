import pandas as pd
import numpy as np
import string
import json
import time

from nltk.tokenize import TweetTokenizer
from tarfile import ExFileObject
from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS
from pandas import DataFrame
from numba import jit

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


def extract_spacy_features(texts, spacy_nlp):
    features["num_emoticons"] = 0
    features["num_contractions"] = 0
    unique_words = set()
    for token in tokenized:
        if token.text in UNICODE_EMOJI:
            features["num_emoticons"] += 1

        if token.text in CONTRACTIONS:
            features["num_contractions"] += 1

        unique_words.add(token.text)

    features["num_unique_words"] = len(unique_words)


@jit(forceobj=True)
def fast_extract_features(texts):
    all_features = []
    
    for text in texts:
        text = str(text)
        features = {}
        tokenized = text.split()

        features["tweet_length"] = len(text)
        features["num_words"] = len(tokenized)
        features["num_exclamation_pts"] = text.count("!")
        features["num_question_mks"] = text.count("?")
        features["num_periods"] = text.count(".")
        features["num_hyphens"] = text.count("-")

        features["num_capitals"] = 0
        features["num_punctuation_mks"] = 0
        for char in text:
            if char.isupper():
                features["num_capitals"] += 1

            if char in string.punctuation:
                features["num_punctuation_mks"] += 1

        all_features.append(features)

    return all_features


@jit(nopython=True)
def language_detect(text, spacy_nlp):
    """
    Check language using spacy_langdetect and return to DataFrame
    """
    lang = spacy_nlp(str(text))._.language
    if lang["score"] >= .9:
        return lang["language"]
    else:
        return "en"
