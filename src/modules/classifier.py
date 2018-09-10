import os

import numpy as np
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, \
    recall_score, f1_score, cohen_kappa_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.modules.stats import save_classifiers_comparison, save_classifier_stats
from src.util.io import write_text, write_data_frame


def run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to, mlp_classifier=True
):
    all_stats = []

    for classifier in get_default_classifiers(mlp_classifier):
        stats = run_classifier(
            classifier, X_train, y_train, X_test, y_test,
            labels, feature_names, df_test, save_to
        )
        all_stats.append(stats)

    abbreviations = get_default_abbreviations(mlp_classifier)
    save_classifiers_comparison(all_stats, abbreviations, save_to)


def get_default_classifiers(mlp_classifier=True):
    # noinspection PyTypeChecker
    return [
        MultinomialNB(),
        LogisticRegression(),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(n_estimators=100),
        AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100),
        BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100),
        LinearSVC(),
    ] + ([MLPClassifier()] if mlp_classifier else [])


def get_default_abbreviations(mlp_classifier=True):
    return ["NB", "LR", "RF", "GB", "AB", "BAG", "SVM"]\
           + (["MLP"] if mlp_classifier else [])


def get_lite_classifiers():
    return [
        MultinomialNB(),
        LogisticRegression(),
        RandomForestClassifier(n_estimators=100),
        LinearSVC()
    ]


def run_classifier(
        classifier, X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
):
    classifier.fit(X_train, y_train)

    training_y_pred = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    precision = precision_score(y_test, y_pred, labels=labels, average=None)
    recall = recall_score(y_test, y_pred, labels=labels, average=None)
    F1 = f1_score(y_test, y_pred, labels=labels, average=None)

    stats = {
        "name": classifier.__class__.__name__,
        "params": classifier.get_params(),
        "labels": labels,
        "training_y_pred": training_y_pred,
        "training_accuracy": accuracy_score(y_train, training_y_pred),
        "training_confusion_matrix":
            confusion_matrix(y_train, training_y_pred, labels),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "precision_mean": np.mean(precision),
        "precision_std": np.std(precision),
        "recall": recall,
        "recall_mean": np.mean(recall),
        "recall_std": np.std(recall),
        "F1": F1,
        "F1_mean": np.mean(F1),
        "F1_std": np.std(F1),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels),
        "top_features": top_features(classifier, feature_names),
        "accurate_rows": accurate_rows(y_test, y_pred, df_test),
        "error_rows": error_rows(y_test, y_pred, df_test)
    }

    if save_to is not None:
        save_classifier_stats(stats, save_to)

    return stats


def top_features(classifier, feature_names):
    weight_classifiers = (LogisticRegression, LinearSVC)
    importance_classifiers = (
        RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    )

    if isinstance(classifier, weight_classifiers):
        return _top_feature_weights(classifier, feature_names)
    elif isinstance(classifier, importance_classifiers):
        return _top_feature_importances(classifier, feature_names)
    else:
        return None


def _top_feature_weights(classifier, feature_names):
    feature_weights = [
        (feature_names[index], weight)
        for index, weight in enumerate(classifier.coef_[0])
    ]
    feature_weights.sort(key=lambda x: x[1])

    min_weights = feature_weights[:10]
    max_weights = feature_weights[:-11:-1]
    return min_weights, max_weights


def _top_feature_importances(classifier, feature_names):
    feature_importances = [
        (feature_names[index], importance)
        for index, importance in enumerate(classifier.feature_importances_)
    ]
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return feature_importances[:20]


def accurate_rows(y_test, y_pred, df_test):
    columns = ["test_key", "result_key", "result_full_description"]
    result = df_test.loc[:, columns]

    result["y_true"] = y_test
    result["y_pred"] = y_pred

    return result[result["y_true"] == result["y_pred"]]


def error_rows(y_test, y_pred, df_test):
    columns = ["test_key", "result_key", "result_full_description"]
    result = df_test.loc[:, columns]

    result["y_true"] = y_test
    result["y_pred"] = y_pred

    return result[result["y_true"] != result["y_pred"]]


def run_k_means(X, df, n_clusters, save_to, output=None):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)

    if output is not None:
        result = ""

    for cluster in range(n_clusters):
        indices = np.where(labels == cluster)[0].tolist()
        df_cluster = df.iloc[indices, :]

        write_data_frame(
            os.path.join(save_to, f"cluster_{cluster}.csv"), df_cluster)

        if output is not None:
            result += f"Cluster {cluster}:\n"
            result += str(df_cluster[output].value_counts())
            result += "\n\n"

    if output is not None:
        write_text(os.path.join(save_to, "summary.txt"), result)
