import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

from src.modules.classifier import run_classifier
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y
from src.util.io import write_matplotlib_figure


def penalty(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    classifiers = [
        LogisticRegression(penalty="l1", dual=False),
        LinearSVC(penalty="l1", dual=False)
    ]
    for classifier in classifiers:
        run_classifier(
            classifier, X_train, y_train, X_test, y_test,
            labels, feature_names, df_test, save_to
        )


def regularization_strength_lr(df, output, save_to):
    Cs = [1 / strength for strength in range(100, 10 - 1, -10)]

    accuracies = _regularization_strength_lr_accuracies(df, output, Cs)
    _regularization_strength_lr_plot(Cs, accuracies, save_to)


def _regularization_strength_lr_accuracies(df, output, Cs):
    accuracies = []

    for C in Cs:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        classifier = LogisticRegression(C=C)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def _regularization_strength_lr_plot(Cs, accuracies, save_to):
    plt.plot(Cs, accuracies, marker="o", markersize=5)

    plt.title("Logistic Regression Regularization Strength Trend")
    plt.xlabel("Regularization strength (C)")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)


def regularization_strength_svm(df, output, save_to):
    Cs = [1 / strength for strength in range(100, 10 - 1, -10)]

    accuracies = _regularization_strength_svm_accuracies(df, output, Cs)
    _regularization_strength_svm_plot(Cs, accuracies, save_to)


def _regularization_strength_svm_accuracies(df, output, Cs):
    accuracies = []

    for C in Cs:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        classifier = LinearSVC(C=C)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def _regularization_strength_svm_plot(Cs, accuracies, save_to):
    plt.plot(Cs, accuracies, marker="o", markersize=5)

    plt.title("Support Vector Machine Regularization Strength Trend")
    plt.xlabel("Regularization strength (C)")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)


def class_weight(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    classifiers = [
        LogisticRegression(class_weight="balanced"),
        RandomForestClassifier(n_estimators=100, class_weight="balanced"),
        LinearSVC(class_weight="balanced")
    ]
    for classifier in classifiers:
        run_classifier(
            classifier, X_train, y_train, X_test, y_test,
            labels, feature_names, df_test, save_to
        )


def multi_class(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    # scikit-learn classifiers use "multi_class='ovr'" by default
    classifiers = [
        LogisticRegression(multi_class="multinomial", solver="newton-cg"),
        LinearSVC(multi_class="crammer_singer")
    ]
    for classifier in classifiers:
        run_classifier(
            classifier, X_train, y_train, X_test, y_test,
            labels, feature_names, df_test, save_to
        )


def sag(df, output, save_to):
    max_iters = list(range(500, 10000 + 1, 500))

    accuracies = _sag_accuracies(df, output, max_iters)
    _sag_plot(max_iters, accuracies, save_to)


def _sag_accuracies(df, output, max_iters):
    accuracies = []

    for max_iter in max_iters:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        classifier = LogisticRegression(solver="sag", max_iter=max_iter)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def _sag_plot(max_iters, accuracies, save_to):
    plt.plot(max_iters, accuracies, marker="o", markersize=5)

    plt.title("LR SAG Max Iterations Trend")
    plt.xlabel("Maximum number of iterations")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)


def n_estimators_rf(df, output, save_to):
    n_estimators_values = list(range(20, 200 + 1, 20))

    accuracies = _n_estimators_rf_accuracies(df, output, n_estimators_values)
    _n_estimators_rf_plot(n_estimators_values, accuracies, save_to)


def _n_estimators_rf_accuracies(df, output, n_estimators_values):
    accuracies = []

    for n_estimators in n_estimators_values:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        classifier = RandomForestClassifier(n_estimators=n_estimators)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def _n_estimators_rf_plot(n_estimators_values, accuracies, save_to):
    plt.plot(n_estimators_values, accuracies, marker="o", markersize=5)

    plt.title("Random Forest Number of Trees Trend")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)


def degree_svm(df, output, save_to):
    degrees = list(range(1, 5 + 1))

    accuracies = _degree_svm_accuracies(df, output, degrees)
    _degree_svm_plot(degrees, accuracies, save_to)


def _degree_svm_accuracies(df, output, degrees):
    accuracies = []

    for degree in degrees:
        df_train, df_test = train_test_split(df, test_size=0.2)

        vectorizer = CountVectorizer()
        X_train, y_train, X_test, y_test, _, _ \
            = to_X_y(vectorizer, df_train, df_test, output)

        classifier = SVC(kernel="poly", degree=degree)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


def _degree_svm_plot(degrees, accuracies, save_to):
    plt.plot(degrees, accuracies, marker="o", markersize=5)

    plt.title("Support Vector Machine Polynomial Degree Trend")
    plt.xlabel("Degree of polynomial kernel")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "figure.png"), plt)
