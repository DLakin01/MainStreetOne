import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

from yellowbrick.classifier import ROCAUC, ConfusionMatrix, ClassificationReport
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import Rank1D

from constants import FILE_PATH

df = pd.read_csv(f"{FILE_PATH}\\ms1_df.csv")
df.set_index("id", inplace=True)

gender_map = {"M": 0, "F": 1}
y = df["gender"].map(gender_map)

df.drop(["gender", "language", "combined_text"], axis=1, inplace=True)
onehot_labels = pd.get_dummies(df, columns=["sentiment", "has_mention", "has_hashtags"]).columns.tolist()

# Scale numeric features
numeric_features = ["num_mentions", "num_hashtags", "num_nouns", "num_pronouns", "num_adjectives", "num_particles",
                    "num_words", "tweet_length", "num_exclamation_pts", "num_question_mks", "num_periods", "num_verbs",
                    "num_hyphens", "num_capitals", "num_emoticons", "num_unique_words", "num_conjunctions",
                    "num_numerals", "num_contractions", "num_adverbs", "num_proper_nouns", "num_punctuation_mks"]
numeric_transformer = Pipeline(steps=[
    ("robust", RobustScaler())
])

categorical_features = ["has_mention", "has_hashtags", "sentiment"]
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(categories="auto"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
X = preprocessor.fit_transform(df)

# feature_rank = Rank1D(features=onehot_labels)
# feature_rank.fit_transform(X, y)
# feature_rank.show()
#
# # Take top features and run RFE
# rank_list = feature_rank.ranks_.tolist()
# features_ranking_dict = {
#     feature_rank.features[i]: rank_list[i]
#     for i in range(len(rank_list))
# }
#
# features_to_keep = [k for k, v in sorted(features_ranking_dict.items(), key=lambda item: item[1]) if v >= 0.85]
# df = df[features_to_keep]
#
# new_preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, features_to_keep)
#     ]
# )
# X = new_preprocessor.fit_transform(df)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gb = GradientBoostingClassifier(
    n_estimators=30,
    random_state=1
)

rfecv = RFECV(estimator=gb, cv=3, scoring="accuracy", verbose=1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# rocauc_viz = ROCAUC(gb, classes=["Male", "Female"])
# rocauc_viz.fit(X_train, y_train)
# rocauc_viz.score(X_test, y_test)
# rocauc_viz.show()

print("ok")

# feature_importance = FeatureImportances(gb, labels=onehot_labels)
# feature_importance.fit(X, y)
# feature_importance.show()
#
# # Remove lowest-performing features and see if we get a better model
# features_to_keep = feature_importance.features_.tolist()[-3:]
# df = df[features_to_keep]
#
# new_preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, features_to_keep)
#     ]
# )
#
# X = new_preprocessor.fit_transform(df)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# gb2 = GradientBoostingClassifier(
#         n_estimators=100,
#         max_depth=4,
#         n_iter_no_change=5,
#         tol=0.01,
#         random_state=1
# )
#
# rocauc_viz = ROCAUC(gb2, classes=["Male", "Female"])
# rocauc_viz.fit(X_train, y_train)
# rocauc_viz.score(X_test, y_test)
# rocauc_viz.show()


# confusion_viz = ConfusionMatrix(gb2, classes=["Male", "Female"])
# confusion_viz.score(X_test, y_test)
# confusion_viz.show()
#
# classification_viz = ClassificationReport(gb2, classes=["Male", "Female"])
# classification_viz.score(X_test, y_test)
# classification_viz.show()

