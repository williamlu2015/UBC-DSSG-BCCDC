import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.manifold import TSNE

from src.modules.vectorizer import vectorize
from src.util.io import write_matplotlib_figure


def pca_2d(df, output, labels, save_to):
    X_r, variances = _pca_2d_transform(df, output)
    _pca_2d_plot(df, output, X_r, variances, labels, save_to)


def _pca_2d_transform(df, output):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X_dense)
    variances = pca.explained_variance_ratio_

    return X_r, variances


def _pca_2d_plot(df, output, X_r, variances, labels, save_to):
    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    for label in labels:
        indices, = np.where(df[output] == label)
        X_curr = X_r[indices, :]

        plt.scatter(X_curr[:, 0], X_curr[:, 1], s=9, label=label)

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    plt.xlabel(
        "Principal Component 1: "
        + "{:.2%}".format(variances[0]) + " variance explained", fontsize=24)
    plt.ylabel(
        "Principal Component 2: "
        + "{:.2%}".format(variances[1]) + " variance explained", fontsize=24)
    plt.title(
        f"Principal Component Analysis (1000 features)",
        fontsize=48)
    plt.legend(markerscale=6, fontsize=36)

    write_matplotlib_figure(os.path.join(save_to, "pca_2d.png"), plt)


def lda_2d(df, output, labels, save_to):
    # WARNING: this trial crashes when the data has only 2 classes.
    X_r, variances = _lda_2d_transform(df, output)
    _lda_2d_plot(df, output, X_r, variances, labels, save_to)


def _lda_2d_transform(df, output):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit_transform(X_dense, y)
    variances = lda.explained_variance_ratio_

    return X_r, variances


def _lda_2d_plot(df, output, X_r, variances, labels, save_to):
    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    for label in labels:
        indices, = np.where(df[output] == label)
        X_curr = X_r[indices, :]

        plt.scatter(X_curr[:, 0], X_curr[:, 1], s=9, label=label)

    plt.xlabel(
        "Attribute 1: " + "{:.2%}".format(variances[0]) + " variance explained",
        fontsize=24)
    plt.ylabel(
        "Attribute 2: " + "{:.2%}".format(variances[1]) + " variance explained",
        fontsize=24)
    plt.title(
        f"Linear Discriminant Analysis (1000 features)",
        fontsize=48)
    plt.legend(markerscale=6, fontsize=36)

    write_matplotlib_figure(os.path.join(save_to, "lda_2d.png"), plt)


def pca_3d(df, output, labels, save_to):
    X_r, variances = _pca_3d_transform(df, output)
    _pca_3d_plot(df, output, X_r, variances, labels, save_to)


def _pca_3d_transform(df, output):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    pca = PCA(n_components=3)
    X_r = pca.fit_transform(X_dense)
    variances = pca.explained_variance_ratio_

    return X_r, variances


def _pca_3d_plot(df, output, X_r, variances, labels, save_to):
    fig = plt.figure()
    ax = Axes3D(fig)

    for label in labels:
        indices, = np.where(df[output] == label)
        X_curr = X_r[indices, :]

        ax.scatter(
            X_curr[:, 0], X_curr[:, 1], zs=X_curr[:, 2], s=1.5, label=label)

    ax.set_xlabel(
        "Principal Component 1: "
        + "{:.2%}".format(variances[0]) + " variance explained")
    ax.set_ylabel(
        "Principal Component 2: "
        + "{:.2%}".format(variances[1]) + " variance explained")
    ax.set_zlabel(
        "Principal Component 3: "
        + "{:.2%}".format(variances[2]) + " variance explained")
    plt.title(f"Principal Component Analysis (1000 features)")
    plt.legend(markerscale=6)

    write_matplotlib_figure(os.path.join(save_to, "pca_3d.png"), plt)


def lda_3d(df, output, labels, save_to):
    # WARNING: this trial crashes when the data has only 3 classes.
    X_r, variances = _lda_3d_transform(df, output)
    _lda_3d_plot(df, output, X_r, variances, labels, save_to)


def _lda_3d_transform(df, output):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    # reduce feature space to prevent MemoryError
    selection = SelectKBest(chi2, 1000)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    lda = LinearDiscriminantAnalysis(n_components=3)
    X_r = lda.fit_transform(X_dense, y)
    variances = lda.explained_variance_ratio_

    return X_r, variances


def _lda_3d_plot(df, output, X_r, variances, labels, save_to):
    fig = plt.figure()
    ax = Axes3D(fig)

    for label in labels:
        indices, = np.where(df[output] == label)
        X_curr = X_r[indices, :]

        ax.scatter(X_curr[:, 0], X_curr[:, 1], zs=X_curr[:, 2], s=1.5, label=label)

    ax.set_xlabel(
        "Attribute 1: " + "{:.2%}".format(variances[0]) + " variance explained")
    ax.set_ylabel(
        "Attribute 2: " + "{:.2%}".format(variances[1]) + " variance explained")
    ax.set_zlabel(
        "Attribute 3: " + "{:.2%}".format(variances[2]) + " variance explained")
    plt.title(f"Linear Discriminant Analysis (1000 features)")
    plt.legend(markerscale=6)

    write_matplotlib_figure(os.path.join(save_to, "lda_3d.png"), plt)


def t_sne(df, output, labels, save_to):
    X_embedded = _t_sne_transform(df, output)
    _t_sne_plot(df, output, X_embedded, labels, save_to)


def _t_sne_transform(df, output):
    vectorizer = CountVectorizer()
    X, _, _ = vectorize(vectorizer, df["result_full_description"])
    y = df[output]

    selection = SelectKBest(chi2, 50)
    X_new = selection.fit_transform(X, y)
    X_dense = X_new.toarray()

    tsne = TSNE(n_components=2, verbose=1)
    X_embedded = tsne.fit_transform(X_dense)

    return X_embedded


def _t_sne_plot(df, output, X_embedded, labels, save_to):
    dpi = 218
    plt.figure(num=1).set_size_inches(5120 / dpi, 2880 / dpi)

    for label in labels:
        indices, = np.where(df[output] == label)
        X_curr = X_embedded[indices, :]

        plt.scatter(X_curr[:, 0], X_curr[:, 1], s=9, label=label)

    plt.title("T-SNE (50 features)", fontsize=48)
    plt.legend(markerscale=6, fontsize=36)

    write_matplotlib_figure(os.path.join(save_to, "t_sne.png"), plt)


def _max_tractable_features(X, y):
    lo = 1
    hi = X.shape[1]
    while True:
        if lo >= hi:
            return _evaluate(X, y, lo)
        else:
            mid = (lo + hi + 1) // 2
            try:
                _evaluate(X, y, mid)
                lo = mid
            except MemoryError:
                hi = mid - 1


def _evaluate(X, y, num_features):
    selection = SelectKBest(chi2, num_features)
    X_new = selection.fit_transform(X, y)
    return X_new.toarray()
