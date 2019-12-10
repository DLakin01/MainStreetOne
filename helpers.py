import pandas as pd
import numpy as np
import string
import json
import time

from langdetect.lang_detect_exception import LangDetectException
from tarfile import ExFileObject
from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS
from numba.typed import Dict, List
from numba import jit, types

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
    all_features = []

    for text in texts:
        text = str(text)
        tokenized = spacy_nlp(text)

        features = {
            "num_words": len(tokenized),
            "num_emoticons": 0,
            "num_contractions": 0,
            "num_adjectives": 0,
            "num_adverbs": 0,
            "num_conjunctions": 0,
            "num_nouns": 0,
            "num_numerals": 0,
            "num_particles": 0,
            "num_pronouns": 0,
            "num_proper_nouns": 0,
            "num_verbs": 0
        }

        unique_words = set()
        for token in tokenized:
            if token._.is_emoji or token.text in UNICODE_EMOJI or token.text in UNICODE_EMOJI_ALIAS:
                features["num_emoticons"] += 1

            if token.text in CONTRACTIONS:
                features["num_contractions"] += 1

            if token.pos_ in POS_MAP:
                column_key = POS_MAP[token.pos_]
                features[column_key] += 1

            unique_words.add(token.text)

        features["num_unique_words"] = len(unique_words)
        all_features.append(features)

    return all_features


@jit(forceobj=True)
def fast_extract_features(texts):
    all_features = []
    
    for text in texts:
        text = str(text)

        features = {
            "tweet_length": len(text),
            "num_exclamation_pts": text.count("!"),
            "num_question_mks": text.count("?"),
            "num_periods": text.count("."),
            "num_hyphens": text.count("-"),
            "num_capitals": 0,
            "num_punctuation_mks": 0
        }

        for char in text:
            if char.isupper():
                features["num_capitals"] += 1

            if char in string.punctuation:
                features["num_punctuation_mks"] += 1

        all_features.append(features)

    return all_features


def language_detect(texts, spacy_nlp):
    """
    Check language using spacy_langdetect and return to DataFrame
    """
    all_languages = []

    for text in texts:
        try:
            text = str(text)
            lang = spacy_nlp(text)._.language
            if lang["score"] >= .9:
                all_languages.append({"language": lang["language"]})
            else:
                all_languages.append({"language": "en"})

        except (TypeError, LangDetectException):
            pass

    return all_languages
