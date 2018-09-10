from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from src.modules.classifier import run_default_classifiers
from src.modules.splitter import downsample
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y


def downsample_train(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train = downsample(df_train, output)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer()
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )


def downsample_train_test(df, output, labels, save_to):
    df = downsample(df, output)
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
