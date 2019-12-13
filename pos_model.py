import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import spacy
import math

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from matplotlib.pylab import rcParams
from spacymoji import Emoji
from sklearn import metrics

from helpers import extract_linguistic_features
from constants import FILE_PATH, SPACY_LANGS


def model_fit(estimator, X, y, predictors):
    # Fit the algorithm on the data
    estimator.fit(X, y)

    # Predict training set
    train_predictions = estimator.predict(X[predictors])
    train_predprob = estimator.predict_proba(X[predictors])[:, 1]

    # Perform cross-validation
    cv_score = cross_val_score(estimator, X[predictors], y, cv=5, scoring="roc_auc")

    # Print model report
    print("\n Model Report")
    print("Accuracy: %.4g" % metrics.accuracy_score(y.values, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, train_predprob))

    print("CV Score: Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
          (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance
    feat_imp = pd.Series(estimator.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind="bar", title="Feature Importances")
    plt.ylabel("Feature Importance Score")


df = pd.read_csv(f"{FILE_PATH}\\ms1_multilang_df.csv")
df.set_index("id", inplace=True)
# df = df.sample(frac=0.2, random_state=0)
#
# # Throw out samples that spaCy can't parse
# multilang_df = df[df["language"].isin(SPACY_LANGS)]
#
# split_by_lang = [{"lang": lang, "df": multilang_df[multilang_df["language"] == lang]} for lang in SPACY_LANGS]
#
# for item in split_by_lang:
#     nlp = spacy.load(item["lang"])
#     emoji = Emoji(nlp, merge_spans=False)
#     nlp.add_pipe(emoji, first=True)
#
#     tweet_ids = item["df"].index.tolist()
#     texts = item["df"]["combined_text"].tolist()
#     if len(texts):
#         spacy_features = extract_linguistic_features(texts, tweet_ids, nlp)
#         temp_df = pd.DataFrame.from_records(spacy_features)
#         temp_df.set_index("id", inplace=True)
#
#         item["df"] = item["df"].join(temp_df)
#
# multilang_df = pd.concat([lang_item["df"] for lang_item in split_by_lang])
# multilang_df.to_csv(f"{FILE_PATH}\\ms1_multilang_df.csv")

# y = np.array(df["gender"].map({"M": 0, "F": 1}))

df.drop(["gender", "language", "combined_text"], axis=1, inplace=True)
onehot_labels = pd.get_dummies(df, columns=["sentiment", "has_mention", "has_hashtags"]).columns.tolist()

# # Scale numeric features
# numeric_features = ["num_mentions", "num_hashtags", "num_nouns", "num_pronouns", "num_adjectives", "num_particles",
#                     "num_words", "tweet_length", "num_exclamation_pts", "num_question_mks", "num_periods", "num_verbs",
#                     "num_hyphens", "num_capitals", "num_emoticons", "num_unique_words", "num_conjunctions",
#                     "num_numerals", "num_contractions", "num_adverbs", "num_proper_nouns", "num_punctuation_mks"]
# numeric_transformer = Pipeline(steps=[
#     ("robust", RobustScaler())
# ])
#
# categorical_features = ["has_mention", "has_hashtags", "sentiment"]
# categorical_transformer = Pipeline(steps=[
#     ("onehot", OneHotEncoder(categories="auto"))
# ])
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features)
#     ]
# )
# X = preprocessor.fit_transform(df)

with open("y_pos.pkl", "rb") as pkl:
    y = pickle.load(pkl)

with open("X_pos.pkl", "rb") as pkl:
    X = pickle.load(pkl)

train_df = pd.DataFrame(X, columns=onehot_labels)
predictors = onehot_labels

# gmb0 = GradientBoostingClassifier(random_state=42)
# model_fit(gmb0, train_df, y, predictors)

# param_test1 = {"n_estimators": range(50, 151, 20)}
# grid_search1 = GridSearchCV(
#     GradientBoostingClassifier(
#         learning_rate=0.1,
#         max_depth=15,
#         max_features="sqrt",
#         subsample=0.8,
#         random_state=42,
#         n_iter_no_change=4,
#         tol=1e-3,
#         verbose=1
#     ),
#     param_grid=param_test1, scoring="roc_auc", cv=3, verbose=2
# ).fit(train_df[predictors], y)
#
# print(grid_search1.cv_results_, grid_search1.best_params_, grid_search1.best_score_)


# param_test2 = {"max_depth": range(5, 16, 2)}
#
# grid_search2 = GridSearchCV(
#     GradientBoostingClassifier(
#         learning_rate=0.15,
#         n_estimators=90,
#         max_features="sqrt",
#         subsample=0.8,
#         n_iter_no_change=4,
#         tol=1e-3,
#         random_state=42,
#         verbose=1
#     ),
#     param_grid=param_test2, scoring="roc_auc", verbose=2, cv=3
# ).fit(train_df[predictors], y)
#
# print(grid_search2.cv_results_, grid_search2.best_params_, grid_search2.best_score_)

param_test3 = {"min_samples_split": range(200, 800, 200)}
grid_search3 = GridSearchCV(
    GradientBoostingClassifier(
        learning_rate=0.15,
        n_estimators=90,
        max_depth=13,
        max_features="sqrt",
        n_iter_no_change=4,
        tol=1e-3,
        subsample=0.8,
        random_state=42,
        verbose=1
    ),
    param_grid=param_test3, scoring="roc_auc", verbose=2, cv=3
).fit(train_df[predictors], y)

print(grid_search3.cv_results_, grid_search3.best_params_, grid_search3.best_score_)

param_test4 = {}

# confusion_viz = ConfusionMatrix(gb2, classes=["Male", "Female"])
# confusion_viz.score(X_test, y_test)
# confusion_viz.show()
#
# classification_viz = ClassificationReport(gb2, classes=["Male", "Female"])
# classification_viz.score(X_test, y_test)
# classification_viz.show()

