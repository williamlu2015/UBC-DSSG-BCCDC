import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

from src.modules.classifier import run_k_means
from src.modules.vectorizer import vectorize
from src.util.io import write_matplotlib_figure


def cluster_labelled(df, output, n_clusters, save_to):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])

    run_k_means(X, df, n_clusters, save_to, output=output)


def cluster_labelled_1000_features(df, output, n_clusters, save_to):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)

    run_k_means(X_new, df, n_clusters, save_to, output=output)


def cluster_labelled_1000_features_pca(df, output, n_clusters, save_to):
    labels, X_r, variances = _pca_transform(df, output, n_clusters)
    _pca_plot(n_clusters, labels, X_r, variances, save_to)


def _pca_transform(df, output, n_clusters):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_new)

    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X_dense)
    variances = pca.explained_variance_ratio_

    return labels, X_r, variances


def _pca_plot(n_clusters, labels, X_r, variances, save_to):
    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    for cluster in range(n_clusters):
        indices = np.where(labels == cluster)[0].tolist()
        X_curr = X_r[indices, :]

        plt.scatter(X_curr[:, 0], X_curr[:, 1], s=9, label=f"Cluster {cluster}")

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    plt.xlabel(
        "Principal Component 1: "
        + "{:.2%}".format(variances[0]) + " variance explained", fontsize=24)
    plt.ylabel(
        "Principal Component 2: "
        + "{:.2%}".format(variances[1]) + " variance explained", fontsize=24)
    plt.title("PCA / K-Means (1000 features)", fontsize=48)
    plt.legend(markerscale=6, fontsize=36)

    write_matplotlib_figure(os.path.join(save_to, "pca_kmeans.png"), plt)


def cluster_labelled_1000_features_lda(df, output, n_clusters, save_to):
    # WARNING: this trial crashes when the data has only 2 classes.
    labels, X_r, variances = _lda_transform(df, output, n_clusters)
    _lda_plot(n_clusters, labels, X_r, variances, save_to)


def _lda_transform(df, output, n_clusters):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_new)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit_transform(X_dense, y)
    variances = lda.explained_variance_ratio_

    return labels, X_r, variances


def _lda_plot(n_clusters, labels, X_r, variances, save_to):
    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    for cluster in range(n_clusters):
        indices = np.where(labels == cluster)[0].tolist()
        X_curr = X_r[indices, :]

        plt.scatter(X_curr[:, 0], X_curr[:, 1], s=9, label=f"Cluster {cluster}")

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    plt.xlabel(
        "Attribute 1: " + "{:.2%}".format(variances[0]) + " variance explained")
    plt.ylabel(
        "Attribute 2: " + "{:.2%}".format(variances[1]) + " variance explained")
    plt.title("LDA / K-Means (1000 features)")
    plt.legend(markerscale=6, fontsize=36)

    write_matplotlib_figure(os.path.join(save_to, "lda_kmeans.png"), plt)


def cluster_labelled_tfidf(df, output, n_clusters, save_to):
    vectorizer = TfidfVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])

    run_k_means(X, df, n_clusters, save_to, output=output)


def cluster_labelled_variance(df, output, n_clusters, save_to):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])

    selection = VarianceThreshold(threshold=0.01)
    X_new = selection.fit_transform(X)

    run_k_means(X_new, df, n_clusters, save_to, output=output)


def cluster_all(df, n_clusters, save_to):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])

    run_k_means(X, df, n_clusters, save_to)
