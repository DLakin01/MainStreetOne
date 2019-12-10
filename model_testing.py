import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import spacy

from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE
from spacy_langdetect import LanguageDetector
from spacymoji import Emoji

from helpers import fast_extract_features, language_detect, extract_spacy_features


def simple_logistic_classify(X_tr, y_tr, X_te, y_te, description):
    m = LogisticRegression().fit(X_tr, y_tr)
    s = m.score(X_te, y_te)
    print('Test score with', description, 'features:', s)
    return m


df = pd.read_csv("C:\\Users\\DLaki\\OneDrive\\Desktop\\Github\\ms1_df.csv")
df = df.sample(frac=0.05, random_state=1)

# Cleaning and feature extraction
nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

# Check language of each tweet and throw out samples that spaCy can't parse
df = df.join(pd.DataFrame.from_records(
    language_detect(df["combined_text"].values, nlp)
))
df = df[~df["language"].isin(["de", "el", "es", "fr", "it", "lt", "nb", "nl", "pt"])]

# Get counts of simple linguistic features
df = df.join(pd.DataFrame.from_records(
    fast_extract_features(df["combined_text"].values)
))

# # Get counts of parts of speech, emoticons, contractions
emoji = Emoji(nlp, merge_spans=False)
nlp.add_pipe(emoji, first=True)

df = df.join(pd.DataFrame.from_records(
    extract_spacy_features(df["combined_text"].values, nlp)
))

df.to_csv("C:\\Users\\DLaki\\OneDrive\\Desktop\\Github\\ms1_df.csv")

# Prepare for modeling
# X = df.drop("gender", axis=1)
#
# gender_map = {"M": 0, "F": 1}
# df["gender"] = df["gender"].map(gender_map)
# y = df["gender"]
#
#
# plt.figure(figsize=(12, 10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, fmt=".2g")
# plt.show()
#
# print("ok")



