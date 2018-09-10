import os

import numpy as np
import pandas as pd

from src.util.increment_tuple import inc_first, inc_zeroth
from src.util.io import write_text, write_data_frame


def save_stats(df, dirname):
    accuracy = find_accuracy(df)
    precision = find_precision(df)
    recall = find_recall(df)
    F1 = find_F1(precision, recall)
    accurate_rows = find_accurate_rows(df)
    error_rows = find_error_rows(df)

    write_text(os.path.join(dirname, "accuracy.txt"), f"{accuracy}\n")
    write_data_frame(os.path.join(dirname, "precision.csv"), precision)
    write_data_frame(os.path.join(dirname, "recall.csv"), recall)
    write_data_frame(os.path.join(dirname, "F1.csv"), F1, na_rep="NaN")
    write_data_frame(os.path.join(dirname, "accurate_rows.csv"), accurate_rows)
    write_data_frame(os.path.join(dirname, "error_rows.csv"), error_rows)


def find_accuracy(df):
    return np.mean(df["is_match"])


def find_precision(df):
    counts = {}
    for _, row in df.iterrows():
        pred_val = row["y_pred"]

        if pred_val not in counts:
            counts[pred_val] = (0, 0)   # (num_relevant, num_selected)

        counts[pred_val] = inc_first(counts[pred_val])
        if row["is_match"]:
            counts[pred_val] = inc_zeroth(counts[pred_val])

    data = []
    for y_true, (num_relevant, num_selected) in counts.items():
        precision = num_relevant / num_selected
        data.append([y_true, num_relevant, num_selected, precision])

    result = pd.DataFrame(data=data, columns=[
        "y_pred", "num_relevant", "num_selected", "precision"
    ])
    result.sort_values("precision", inplace=True)
    return result


def find_recall(df):
    counts = {}
    for _, row in df.iterrows():
        true_val = row["y_true"]

        if true_val not in counts:
            counts[true_val] = (0, 0)   # (num_selected, num_relevant)

        counts[true_val] = inc_first(counts[true_val])
        if row["is_match"]:
            counts[true_val] = inc_zeroth(counts[true_val])

    data = []
    for y_true, (num_selected, num_relevant) in counts.items():
        recall = num_selected / num_relevant
        data.append([y_true, num_selected, num_relevant, recall])

    result = pd.DataFrame(data=data, columns=[
        "y_true", "num_selected", "num_relevant", "recall"
    ])
    result.sort_values("recall", inplace=True)
    return result


def find_F1(precision, recall):
    precision = precision.drop(
        columns=["num_relevant", "num_selected"], inplace=False)
    precision.rename(columns={"y_pred": "y"}, inplace=True)

    recall = recall.drop(
        columns=["num_selected", "num_relevant"], inplace=False)
    recall.rename(columns={"y_true": "y"}, inplace=True)

    result = precision.merge(recall, on=["y"], how="outer")
    result["F1"] = result.apply(
        lambda row: _compute_F1(row["precision"], row["recall"]), axis=1
    )
    result.sort_values("F1", inplace=True)
    return result


def _compute_F1(precision_val, recall_val):
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def find_accurate_rows(df):
    return df[df["is_match"]]


def find_error_rows(df):
    return df[~df["is_match"]]
