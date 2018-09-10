from sklearn import clone
from sklearn.model_selection import KFold

from src.modules.classifier import run_classifier
from src.modules.feature_selector import select
from src.modules.vectorizer import to_X_y


def cv_vectorizer(df, output, vectorizers, classifier):
    all_stats = [[] for _ in vectorizers]

    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(df):
        df_train = df.iloc[train_indices, :]
        df_test = df.iloc[test_indices, :]

        for index, vectorizer in enumerate(vectorizers):
            X_train, y_train, X_test, y_test, feature_names, _ \
                = to_X_y(vectorizer, df_train, df_test, output)

            _classifier = clone(classifier)
            stats = run_classifier(
                _classifier, X_train, y_train, X_test, y_test,
                None, feature_names, df_test, None
            )

            all_stats[index].append(stats)

    return all_stats


def cv_classifier(df, output, vectorizer, classifiers):
    all_stats = [[] for _ in classifiers]

    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(df):
        df_train = df.iloc[train_indices, :]
        df_test = df.iloc[test_indices, :]

        _vectorizer = clone(vectorizer)
        X_train, y_train, X_test, y_test, feature_names, _\
            = to_X_y(_vectorizer, df_train, df_test, output)

        for index, classifier in enumerate(classifiers):
            stats = run_classifier(
                classifier, X_train, y_train, X_test, y_test,
                None, feature_names, df_test, None
            )

            all_stats[index].append(stats)

    return all_stats


def cv_selection(df, output, vectorizer, classifier, selections):
    all_stats = [[] for _ in selections]

    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(df):
        df_train = df.iloc[train_indices, :]
        df_test = df.iloc[test_indices, :]

        _vectorizer = clone(vectorizer)
        X_train, y_train, X_test, y_test, feature_names, _ \
            = to_X_y(_vectorizer, df_train, df_test, output)

        for index, selection in enumerate(selections):
            X_train_new, X_test_new, feature_names_new\
                = select(selection, X_train, y_train, X_test, feature_names)

            _classifier = clone(classifier)
            stats = run_classifier(
                _classifier, X_train_new, y_train, X_test_new, y_test,
                None, feature_names_new, df_test, None
            )

            all_stats[index].append(stats)

    return all_stats
