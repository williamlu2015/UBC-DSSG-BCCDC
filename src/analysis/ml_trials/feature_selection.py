import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.modules.classifier import run_default_classifiers, get_lite_classifiers
from src.modules.feature_selector import select
from src.modules.stats import save_train_test, save_X_y, save_filtered_features
from src.modules.vectorizer import to_X_y
from src.util.io import write_matplotlib_figure


def feature_selection_variance(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)

    selection = VarianceThreshold(threshold=0.01)
    X_train_new, X_test_new, feature_names_new \
        = select(selection, X_train, y_train, X_test, feature_names)
    save_X_y(X_train_new, y_train, X_test_new, y_test, feature_names_new, save_to)

    save_filtered_features(feature_names, feature_names_new, save_to)

    run_default_classifiers(
        X_train_new, y_train, X_test_new, y_test,
        labels, feature_names_new, df_test, save_to
    )


def feature_selection_variance_trend(df, output, save_to):
    thresholds = [0.01 * i for i in range(1, 20 + 1)]

    accuracies = _feature_selection_variance_trend_accuracies(
        df, output, thresholds)
    _feature_selection_variance_trend_plot(thresholds, accuracies, save_to)


def _feature_selection_variance_trend_accuracies(df, output, thresholds):
    classifiers = get_lite_classifiers()
    accuracies = [[] for _ in classifiers]

    for threshold in thresholds:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, feature_names, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        selection = VarianceThreshold(threshold=threshold)
        X_train_new, X_test_new, _ \
            = select(selection, X_train, y_train, X_test, feature_names)

        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train_new, y_train)
            y_pred = classifier.predict(X_test_new)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies[index].append(accuracy)

    return accuracies


def _feature_selection_variance_trend_plot(thresholds, accuracies, save_to):
    labels = [
        "Naive Bayes", "Logistic Regression", "Random Forest (100 trees)",
        "Support Vector Machine (Linear)"
    ]
    assert len(labels) == len(accuracies)

    handles = []
    for accuracy, label in zip(accuracies, labels):
        line, = plt.plot(
            thresholds, accuracy, marker="o", markersize=5, label=label)
        handles.append(line)

    plt.title("Variance Threshold Trend")
    plt.xlabel("Variance threshold")
    plt.ylabel("Accuracy")
    plt.legend(handles=handles)

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)


def feature_selection_chi2(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)

    selection = SelectKBest(chi2, 100)
    X_train_new, X_test_new, feature_names_new\
        = select(selection, X_train, y_train, X_test, feature_names)
    save_X_y(X_train_new, y_train, X_test_new, y_test, feature_names_new, save_to)

    save_filtered_features(feature_names, feature_names_new, save_to)

    run_default_classifiers(
        X_train_new, y_train, X_test_new, y_test,
        labels, feature_names_new, df_test, save_to
    )


def feature_selection_chi2_trend(df, output, save_to):
    ks = list(range(10, 200 + 1, 10))

    accuracies = _feature_selection_chi2_trend_accuracies(df, output, ks)
    _feature_selection_chi2_trend_plot(ks, accuracies, save_to)


def _feature_selection_chi2_trend_accuracies(df, output, ks):
    classifiers = get_lite_classifiers()
    accuracies = [[] for _ in classifiers]

    for k in ks:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, feature_names, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        selection = SelectKBest(chi2, k)
        X_train_new, X_test_new, _ \
            = select(selection, X_train, y_train, X_test, feature_names)

        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train_new, y_train)
            y_pred = classifier.predict(X_test_new)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies[index].append(accuracy)

    return accuracies


def _feature_selection_chi2_trend_plot(ks, accuracies, save_to):
    labels = [
        "Naive Bayes", "Logistic Regression", "Random Forest (100 trees)",
        "Support Vector Machine (Linear)"
    ]
    assert len(labels) == len(accuracies)

    handles = []
    for accuracy, label in zip(accuracies, labels):
        line, = plt.plot(ks, accuracy, marker="o", markersize=5, label=label)
        handles.append(line)

    plt.title("Chi2 trend")
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.legend(handles=handles)

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)
