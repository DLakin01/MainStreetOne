import string
import json

from nltk.tokenize import word_tokenize
from tarfile import ExFileObject

from constants import PUNCTUATION_TABLE, STOP_WORDS, CONTRACTION_MAP


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


def clean_text(text: str) -> list:
    # To lowercase and replace newline characters
    text = text.lower().replace("\n", " ")
    text_list = word_tokenize(text)

    # Strip out stop words, numeric strings, empty strings, and punctuation
    parsed_text = []
    for word in text_list:
        if word in STOP_WORDS:
            continue
        elif word is "":
            continue
        elif word in string.punctuation:
            continue
        elif any(char.isdigit() for char in word):
            continue

        if word in CONTRACTION_MAP:
            parsed_text.append(CONTRACTION_MAP[word])

        elif "/" in word:
            split_words = word.split("/")
            split_words = [w.translate(PUNCTUATION_TABLE) for w in split_words]
            parsed_text.extend(split_words)

        elif "." in word:
            split_words = word.split(".")
            split_words = [w.translate(PUNCTUATION_TABLE) for w in split_words]
            parsed_text.extend(split_words)

        else:
            parsed_text.append(word.translate(PUNCTUATION_TABLE))

    parsed_text = [word for word in parsed_text if len(word) > 1]

    return list(set(parsed_text))