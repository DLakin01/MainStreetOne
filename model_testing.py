import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import spacy

from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE
from spacymoji import Emoji

from helpers import extract_linguistic_features
from constants import FILE_PATH, SPACY_LANGS


def simple_logistic_classify(X_tr, y_tr, X_te, y_te, description):
    m = LogisticRegression().fit(X_tr, y_tr)
    s = m.score(X_te, y_te)
    print('Test score with', description, 'features:', s)
    return m


df = pd.read_csv(f"{FILE_PATH}\\ms1_df.csv", dtype={"combined_text": str})
df = df.sample(frac=0.05, random_state=1)

# Cleaning and feature extraction

# Throw out samples that spaCy can't parse
df = df[df["language"].isin(SPACY_LANGS)]

split_by_lang = [{"lang": lang, "df": df[df["language"] == lang]} for lang in SPACY_LANGS]

for item in split_by_lang:
    nlp = spacy.load(item["lang"])
    emoji = Emoji(nlp, merge_spans=False)
    nlp.add_pipe(emoji, first=True)

    texts = item["df"]["combined_text"].tolist()
    spacy_features = extract_linguistic_features(texts, nlp)

    temp_df = pd.DataFrame.from_records(spacy_features)

    item["df"] = item["df"].join(pd.DataFrame.from_records(spacy_features))

df = pd.concat([lang_item["df"] for lang_item in split_by_lang])


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



