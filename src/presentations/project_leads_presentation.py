import os

import matplotlib.pyplot as plt
import random
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from root import from_root
from src.modules.classifier import top_features
from src.modules.db import extract
from src.modules.vectorizer import to_X_y
from src.util.io import write_matplotlib_figure
from src.visualizations.word_cloud import word_cloud


SAVE_TO = from_root("images\\project_leads_presentation")


def main():
    create_dataset_word_cloud()
    create_logistic_regression_word_clouds()
    create_random_forest_bar_graphs()


def create_dataset_word_cloud():
    df = extract(from_root("sql\\dataset.sql"))

    raw_documents = df["result_full_description"]
    documents = [document.lower() for document in raw_documents]
    word_cloud(documents, output_filename=os.path.join(SAVE_TO, "dataset.png"))


def create_logistic_regression_word_clouds():
    df = extract(from_root("sql\\test_outcome\\2_class.sql"))
    df_train, df_test = train_test_split(df, test_size=0.2)

    vectorizer = CountVectorizer()
    X_train, y_train, _, _, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, "test_outcome")

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    min_weights, max_weights = top_features(classifier, feature_names)

    min_weights_dict = {x[0]: abs(x[1]) for x in min_weights}
    word_cloud(min_weights_dict, max_words=20, stopwords=None,
               max_font_size=900, color_func=_color_red,
               output_filename=os.path.join(
                   SAVE_TO, "logistic_regression_min.png"))

    max_weights_dict = {x[0]: x[1] ** 3 for x in max_weights}
    word_cloud(max_weights_dict, max_words=20, stopwords=None,
               max_font_size=900, color_func=_color_blue,
               output_filename=os.path.join(
                   SAVE_TO, "logistic_regression_max.png"))


def _color_red(*args, **kwargs):
    colors = ["Crimson", "DarkRed", "Red"]
    return random.choice(colors)


def _color_blue(*args, **kwargs):
    colors = ["Blue", "DarkBlue", "DarkSlateBlue", "MediumBlue", "MidnightBlue"]
    return random.choice(colors)


def create_random_forest_bar_graphs():
    weights = _random_forest_weights()
    _random_forest_plot(weights)


def _random_forest_weights():
    df = extract(from_root("sql\\test_outcome\\2_class.sql"))
    df_train, df_test = train_test_split(df, test_size=0.2)

    vectorizer = CountVectorizer()
    X_train, y_train, _, _, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, "test_outcome")

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    weights = top_features(classifier, feature_names)[:10]
    return weights


def _random_forest_plot(weights):
    X = [x[0] for x in weights]
    Y = [x[1] for x in weights]

    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    n = len(X)
    plt.bar(range(n), Y, align="center", color="orange")

    plt.yticks(fontsize=36)
    plt.xticks(range(n), X, rotation=30, fontsize=36)

    plt.title(
        "10 most important features (ranked by Random Forests)", fontsize=48)
    plt.xlabel("Feature names", fontsize=36)
    plt.ylabel("Feature importances", fontsize=36)

    plt.tight_layout()

    write_matplotlib_figure(os.path.join(SAVE_TO, "random_forest.png"), plt)


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    main()

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
