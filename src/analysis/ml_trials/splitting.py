import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from src.modules.classifier import run_default_classifiers
from src.modules.splitter import split_by_index
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y


def split_skewed(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.99)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_organism(df, labels, save_to):
    def organism_splitter(index):
        if df.ix[index, "test_outcome"] == "negative":
            return random.randint(0, 1) == 1
        else:
            return df.ix[index, "level_1"] == "hepatitis c virus"

    df_train, df_test = split_by_index(df, organism_splitter)
    _run_split_trial(df_train, df_test, "test_outcome", labels, save_to)


def split_by_date_half(df, output, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    df.sort_values(by=["result_date_key"])

    m, _ = df.shape
    mid = m // 2
    df_train = df.iloc[0:mid, :]
    df_test = df.iloc[mid:, :]
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to, mlp_classifier=False
    )


def split_by_date_quarter(df, output, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    df.sort_values(by=["result_date_key"])

    m, _ = df.shape
    df_train = df.iloc[0:(m // 4), :]
    df_test = df.iloc[(3 * m // 4):, :]
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to, mlp_classifier=False
    )


def split_by_dss_2_3(df, output, labels, save_to):
    def dss_2_3_splitter(index):
        if df.ix[index, "dss_lis_instance_id"] == 2:
            return False
        elif df.ix[index, "dss_lis_instance_id"] == 3:
            return True
        else:
            raise ValueError

    df_train, df_test = split_by_index(df, dss_2_3_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_dss_3_2(df, output, labels, save_to):
    def dss_3_2_splitter(index):
        if df.ix[index, "dss_lis_instance_id"] == 3:
            return False
        elif df.ix[index, "dss_lis_instance_id"] == 2:
            return True
        else:
            raise ValueError

    df_train, df_test = split_by_index(df, dss_3_2_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_test_type_culture_train(df, output, labels, save_to):
    def culture_train_splitter(index):
        return df.ix[index, "test_type"] != "culture"

    df_train, df_test = split_by_index(df, culture_train_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_test_type_culture_test(df, output, labels, save_to):
    def culture_test_splitter(index):
        return df.ix[index, "test_type"] == "culture"

    df_train, df_test = split_by_index(df, culture_test_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_test_type_antibody_nat(df, output, labels, save_to):
    def antibody_nat_splitter(index):
        if df.ix[index, "test_type"] == "antibody":
            return False
        elif df.ix[index, "test_type"] == "nat/pcr":
            return True
        else:
            raise ValueError

    df_train, df_test = split_by_index(df, antibody_nat_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def split_by_test_type_nat_antibody(df, output, labels, save_to):
    def nat_antibody_splitter(index):
        if df.ix[index, "test_type"] == "nat/pcr":
            return False
        elif df.ix[index, "test_type"] == "antibody":
            return True
        else:
            raise ValueError

    df_train, df_test = split_by_index(df, nat_antibody_splitter)
    _run_split_trial(df_train, df_test, output, labels, save_to)


def _run_split_trial(df_train, df_test, output, labels, save_to):
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )
