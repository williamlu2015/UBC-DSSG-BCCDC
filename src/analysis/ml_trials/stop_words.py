import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.modules.classifier import run_default_classifiers, \
    get_lite_classifiers
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y
from src.util.io import write_matplotlib_figure


def stop_words_english(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer(stop_words="english")
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )


def stop_words_hepatitis(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer(stop_words=["hbsag", "hbv"])
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )


def stop_words_min_df(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer(min_df=5)
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )


def stop_words_min_df_trend(df, output, save_to):
    min_dfs = list(range(5, 100 + 1, 5))

    accuracies = _stop_words_min_df_trend(df, output, min_dfs)
    _stop_words_min_df_trend_plot(min_dfs, accuracies, save_to)


def _stop_words_min_df_trend(df, output, min_dfs):
    classifiers = get_lite_classifiers()
    accuracies = [[] for _ in classifiers]

    for min_df in min_dfs:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer(min_df=min_df)
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies[index].append(accuracy)

    return accuracies


def _stop_words_min_df_trend_plot(min_dfs, accuracies, save_to):
    labels = [
        "Naive Bayes", "Logistic Regression", "Random Forest (100 trees)",
        "Support Vector Machine (Linear)"
    ]
    assert len(labels) == len(accuracies)

    handles = []
    for accuracy, label in zip(accuracies, labels):
        line, = plt.plot(
            min_dfs, accuracy, marker="o", markersize=5, label=label
        )
        handles.append(line)

    plt.title("Minimum Document Frequency Trend")
    plt.xlabel("Minimum document frequency")
    plt.ylabel("Accuracy")
    plt.legend(handles=handles)

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)
