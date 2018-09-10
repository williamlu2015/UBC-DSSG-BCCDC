import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.modules.classifier import error_rows
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y
from src.util.io import write_data_frame


def comparison_lr_rf_intersection(df, output, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    # ===================
    # logistic regression

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    lr_errors = error_rows(y_test, y_pred, df_test)
    lr_errors.rename(columns={"y_pred": "y_pred_lr"}, inplace=True)

    # =============
    # random forest

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    rf_errors = error_rows(y_test, y_pred, df_test)
    rf_errors = rf_errors.drop(columns=["result_full_description", "y_true"])
    rf_errors.rename(columns={"y_pred": "y_pred_rf"}, inplace=True)

    # ================

    result = lr_errors.merge(
        rf_errors, on=["test_key", "result_key"], how="inner"
    )
    write_data_frame(os.path.join(save_to, "errors.csv"), result)


def comparison_rf_svm_intersection(df, output, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    # =============
    # random forest

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    rf_errors = error_rows(y_test, y_pred, df_test)
    rf_errors.rename(columns={"y_pred": "y_pred_rf"}, inplace=True)

    # ======================
    # support vector machine

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    svm_errors = error_rows(y_test, y_pred, df_test)
    svm_errors = svm_errors.drop(columns=["result_full_description", "y_true"])
    svm_errors.rename(columns={"y_pred": "y_pred_svm"}, inplace=True)

    # ================

    result = rf_errors.merge(
        svm_errors, on=["test_key", "result_key"], how="inner"
    )
    write_data_frame(os.path.join(save_to, "errors.csv"), result)


def comparison_lr_rf_difference(df, output, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    # ===================
    # logistic regression

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    lr_errors = error_rows(y_test, y_pred, df_test)
    lr_errors.rename(columns={"y_pred": "y_pred_lr"}, inplace=True)

    # =============
    # random forest

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    rf_errors = error_rows(y_test, y_pred, df_test)
    rf_errors = rf_errors.drop(columns=["result_full_description", "y_true"])
    rf_errors.rename(columns={"y_pred": "y_pred_rf"}, inplace=True)

    # ================

    result = lr_errors.merge(
        rf_errors, on=["test_key", "result_key"], how="left"
    )
    result = result.ix[result["y_pred_rf"].isnull()]

    write_data_frame(os.path.join(save_to, "errors.csv"), result)


def comparison_all_intersection(df, output, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    classifiers = [
        MultinomialNB(), LogisticRegression(),
        RandomForestClassifier(n_estimators=100), LinearSVC()
    ]
    names = ["nb", "lr", "rf", "svm"]

    error_dfs = []
    for classifier, name in zip(classifiers, names):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        error_df = error_rows(y_test, y_pred, df_test)
        error_df.rename(columns={"y_pred": f"y_pred_{name}"}, inplace=True)

        error_dfs.append(error_df)

    result = error_dfs[0]
    for i in range(1, len(error_dfs)):
        error_df = error_dfs[i].drop(
            columns={"result_full_description", "y_true"})
        result = result.merge(
            error_df, on=["test_key", "result_key"], how="inner")

    write_data_frame(os.path.join(save_to, "errors.csv"), result)
