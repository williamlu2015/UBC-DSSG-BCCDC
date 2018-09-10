import os

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.modules.classifier import run_default_classifiers, \
    get_lite_classifiers
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y
from src.util.io import write_matplotlib_figure


def baseline(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )


def data_size(df, output, labels, size, save_to):
    df_sampled = df.sample(size)
    baseline(df_sampled, output, labels, save_to)


def data_size_trend(df, output, save_to):
    sizes = list(range(50, 5000 + 1, 50))

    accuracies = _data_size_trend_accuracies(df, output, sizes)
    _data_size_trend_plot(sizes, accuracies, save_to)


def _data_size_trend_accuracies(df, output, sizes):
    classifiers = get_lite_classifiers()
    accuracies = [[] for _ in classifiers]

    for size in sizes:
        df_curr = df.sample(size)
        df_train, df_test = train_test_split(df_curr, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies[index].append(accuracy)

    return accuracies


def _data_size_trend_plot(sizes, accuracies, save_to):
    labels = [
        "Naive Bayes", "Logistic Regression", "Random Forest (100 trees)",
        "Support Vector Machine (Linear)"
    ]
    assert len(labels) == len(accuracies)

    handles = []
    for accuracy, label in zip(accuracies, labels):
        line, = plt.plot(sizes, accuracy, marker="o", markersize=5, label=label)
        handles.append(line)

    plt.title("Data Size Trend")
    plt.xlabel("Data size (number of rows)")
    plt.ylabel("Accuracy")
    plt.legend(handles=handles)

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)
