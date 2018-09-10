import os
import pprint

import nltk
from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.modules.classifier import run_default_classifiers, run_classifier, \
    top_features
from src.modules.stats import save_train_test, save_X_y
from src.modules.vectorizer import to_X_y, vectorize
from src.util.io import write_text


def ngram(df, output, labels, ngram_range, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to, mlp_classifier=False
    )


def tfidf(df, output, labels, save_to):
    vectorizer = TfidfVectorizer()
    _run_vectorizing_trial(vectorizer, df, output, labels, save_to)


def stemming(df, output, labels, save_to):
    class Tokenizer:
        def __init__(self):
            self.stemmer = SnowballStemmer("english")

        def __call__(self, doc):
            return [self.stemmer.stem(t) for t in word_tokenize(doc)]

    vectorizer = CountVectorizer(tokenizer=Tokenizer())
    _run_vectorizing_trial(vectorizer, df, output, labels, save_to)


def lemmatization(df, output, labels, save_to):
    nltk.download("wordnet")

    class Tokenizer:
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    vectorizer = CountVectorizer(tokenizer=Tokenizer())
    _run_vectorizing_trial(vectorizer, df, output, labels, save_to)


def character_ngrams(df, output, labels, save_to):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
    _run_vectorizing_trial(vectorizer, df, output, labels, save_to)


def character_trigrams(df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    classifier = LinearSVC()
    run_classifier(
        classifier, X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )

    min_weights, max_weights = top_features(classifier, feature_names)
    min_features = [x[0] for x in min_weights]
    max_features = [x[0] for x in max_weights]

    vectorizer = CountVectorizer()
    _, words, _ = vectorize(vectorizer, df_train["result_full_description"])

    min_words = {
        feature: [word for word in words if feature in word]
        for feature in min_features
    }
    max_words = {
        feature: [word for word in words if feature in word]
        for feature in max_features
    }

    write_text(
        os.path.join(save_to, "min_words.txt"),
        pprint.pformat(min_words))
    write_text(
        os.path.join(save_to, "max_words.txt"),
        pprint.pformat(max_words))


def _run_vectorizing_trial(vectorizer, df, output, labels, save_to):
    df_train, df_test = train_test_split(df, test_size=0.2)
    save_train_test(df_train, df_test, save_to)

    X_train, y_train, X_test, y_test, feature_names, _ \
        = to_X_y(vectorizer, df_train, df_test, output)
    save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to)

    run_default_classifiers(
        X_train, y_train, X_test, y_test,
        labels, feature_names, df_test, save_to
    )
