import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pickle
import spacy
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from itertools import product
from spacymoji import Emoji
from sklearn import metrics
from pprint import pprint

from helpers import extract_linguistic_features
from constants import FILE_PATH, SPACY_LANGS


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


df = pd.read_csv(f"{FILE_PATH}\\ms1_df.csv")
df.set_index("id", inplace=True)

# Throw out samples that spaCy can't parse
df = df[df["language"].isin(SPACY_LANGS)]

df = df.sample(n=70000, random_state=42)

split_by_lang = [{"lang": lang, "df": df[df["language"] == lang]} for lang in SPACY_LANGS]

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

X = pd.concat([lang_item["df"] for lang_item in split_by_lang])
y = np.array(X["gender"].map({"M": 0, "F": 1}))
X.to_csv(f"{FILE_PATH}\\ms1_multilang_df.csv")

X.drop(columns=["gender", "language", "combined_text"], inplace=True)
predictors = pd.get_dummies(X, columns=["sentiment", "has_mention", "has_hashtags"]).columns.tolist()

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
X = preprocessor.fit_transform(X)

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


# random_df = pd.DataFrame(columns=list(random_search_params.keys()))
# for i, params in enumerate(random_results["params"]):
#     random_df = random_df.append(pd.DataFrame(params), ignore_index=True)
#
#
# random_df["iteration"] = random_results["iteration"]
# random_df["score"] = random_results["score"]

# confusion_viz = ConfusionMatrix(gb2, classes=["Male", "Female"])
# confusion_viz.score(X_test, y_test)
# confusion_viz.show()

# classification_viz = ClassificationReport(gb2, classes=["Male", "Female"])
# classification_viz.score(X_test, y_test)
# classification_viz.show()
