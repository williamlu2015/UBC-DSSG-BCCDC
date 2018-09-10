import matplotlib.pyplot as plt
import os
import pprint
from math import inf

import numpy as np

from src.util.io import write_data_frame, write_text, write_npz, \
    write_matplotlib_figure


def save_train_test(df_train, df_test, save_to):
    write_data_frame(os.path.join(save_to, "df_train.csv"), df_train)
    write_data_frame(os.path.join(save_to, "df_test.csv"), df_test)


def save_X_y(X_train, y_train, X_test, y_test, feature_names, save_to):
    write_text(
        os.path.join(save_to, "X_y_stats.txt"),

        "X_train shape: " + str(X_train.shape) + "\n"
        + "y_train length: " + str(len(y_train)) + "\n"
        + "X_test shape: " + str(X_test.shape) + "\n"
        + "y_test length: " + str(len(y_test)) + "\n"
        + "Feature names:\n" + pprint.pformat(feature_names) + "\n"
    )
    write_npz(os.path.join(save_to, "X_train.npz"), X_train)
    write_npz(os.path.join(save_to, "X_test.npz"), X_test)


def save_filtered_features(feature_names, feature_names_new, save_to):
    old_set = set(feature_names)
    new_set = set(feature_names_new)

    filtered_features = sorted(old_set.difference(new_set))

    write_text(
        os.path.join(save_to, "filtered_features.txt"),
        pprint.pformat(filtered_features)
    )


def save_classifier_stats(stats, save_to):
    np.set_printoptions(linewidth=inf, formatter={
        "float": lambda x: "{0:0.4f}".format(x)
    })

    write_text(
        os.path.join(save_to, stats["name"] + "_summary.txt"),

        "Classifier: " + stats["name"] + "\n"
        + "Params: " + str(stats["params"]) + "\n"
        + "\n"
        + "Labels: " + str(stats["labels"]) + "\n"
        + "\n"
        + "Training accuracy: " + str(stats["training_accuracy"]) + "\n"
        + "Training confusion matrix:\n"
        + np.array2string(stats["training_confusion_matrix"]) + "\n"
        + "\n"
        + "Accuracy: " + str(stats["accuracy"]) + "\n"
        + "\n"
        + "Precision: " + np.array2string(stats["precision"]) + "\n"
        + "Mean: " + str(stats["precision_mean"]) + "\n"
        + "Std: " + str(stats["precision_std"]) + "\n"
        + "\n"
        + "Recall: " + np.array2string(stats["recall"]) + "\n"
        + "Mean: " + str(stats["recall_mean"]) + "\n"
        + "Std: " + str(stats["recall_std"]) + "\n"
        + "\n"
        + "F1 score: " + np.array2string(stats["F1"]) + "\n"
        + "Mean: " + str(stats["F1_mean"]) + "\n"
        + "Std: " + str(stats["F1_std"]) + "\n"
        + "\n"
        + "Cohen Kappa score: " + str(stats["cohen_kappa"]) + "\n"
        + "Confusion matrix:\n"
        + np.array2string(stats["confusion_matrix"]) + "\n"
        + "\n"
    )

    write_text(
        os.path.join(save_to, stats["name"] + "_top_features.txt"),
        top_features_to_string(stats["top_features"])
    )

    write_data_frame(
        os.path.join(save_to, stats["name"] + "_accurate_rows.csv"),
        stats["accurate_rows"]
    )

    write_data_frame(
        os.path.join(save_to, stats["name"] + "_error_rows.csv"),
        stats["error_rows"]
    )


def top_features_to_string(features):
    if isinstance(features, tuple):
        return "Negative features:\n" + pprint.pformat(features[0]) + "\n" \
               + "Positive features:\n" + pprint.pformat(features[1]) + "\n"
    elif isinstance(features, list):
        return "Top features:\n" + pprint.pformat(features) + "\n"
    elif features is None:
        return "Top features: " + str(None) + "\n"
    else:
        raise ValueError


def save_classifiers_comparison(all_stats, abbreviations, save_to):
    accuracies = [stats["accuracy"] for stats in all_stats]
    n = len(accuracies)

    plt.bar(np.arange(n), accuracies)

    plt.xticks(range(n), abbreviations)
    plt.yticks()

    plt.title("Classifier Comparison")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")

    write_matplotlib_figure(os.path.join(save_to, "comparison.png"), plt)
