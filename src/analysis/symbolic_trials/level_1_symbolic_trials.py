import json

from src.analysis.symbolic_trials.stats import save_stats
from src.util.get_one import get_one


# ==============================================================================
# baseline trial (dumb algorithm, no hardcoding of specific cases)


def baseline(df, save_to):
    df = df.copy()   # don't mutate the original DataFrame
    df.rename(columns={"level_1": "y_true"}, inplace=True)

    _baseline_predict(df)
    _baseline_match(df)

    save_stats(df, save_to)


def _baseline_predict(df):
    df["y_pred"] = df.apply(_baseline_predict_row, axis=1)


def _baseline_predict_row(row):
    candidates_str = row["candidates"]
    candidates_set = set(json.loads(candidates_str))

    if not candidates_set:
        return "*not found"   # candidates_set is empty

    organism = get_one(candidates_set)
    level_1 = organism.split()[0]
    return level_1


def _baseline_match(df):
    df["is_match"] = df.apply(_baseline_match_row, axis=1)


def _baseline_match_row(row):
    return row["y_true"] == row["y_pred"]


# ==============================================================================
# better baseline (smarter algorithm, but still no hardcoding of specific cases)


def better_baseline(df, save_to):
    df = df.copy()  # don't mutate the original DataFrame
    df.rename(columns={"level_1": "y_true"}, inplace=True)

    _preprocess_candidates(df)
    _preprocess_labels(df)

    _better_predict(df)
    _better_match(df)

    save_stats(df, save_to)


def _preprocess_candidates(df):
    df["candidates_prep"] = df.apply(_preprocess_candidates_row, axis=1)


def _preprocess_candidates_row(row):
    candidates_str = row["candidates"]
    candidates_list = json.loads(candidates_str)

    result = set()
    for candidate in candidates_list:
        result.update(candidate.split(", "))
    result = list(result)

    return json.dumps(result)


def _preprocess_labels(df):
    df["y_true_prep"] = df.apply(_preprocess_labels_row, axis=1)


def _preprocess_labels_row(row):
    result = row["y_true"].split(" or ")
    return json.dumps(result)


def _better_predict(df):
    df["y_pred"] = df.apply(_better_predict_row, axis=1)


def _better_predict_row(row):
    candidates_str = row["candidates_prep"]
    candidates_set = set(json.loads(candidates_str))

    if not candidates_set:
        return "*not found"  # candidates_set is empty

    organism = get_one(candidates_set)
    level_1 = organism.split()[0]
    return level_1


def _better_match(df):
    df["is_match"] = df.apply(_better_match_row, axis=1)


def _better_match_row(row):
    true_labels = json.loads(row["y_true_prep"])
    return row["y_pred"] in true_labels


# ==============================================================================
# heuristical trial (smarter algorithm with hardcoding of specific cases)


def heuristical(df, save_to):
    df = df.copy()   # don't mutate the original DataFrame
    df.rename(columns={"level_1": "y_true"}, inplace=True)

    _heuristical_preprocess_candidates(df)
    _heuristical_preprocess_labels(df)

    _heuristical_predict(df)
    _heuristical_match(df)

    save_stats(df, save_to)


def _heuristical_preprocess_candidates(df):
    df["candidates_prep"] = df.apply(
        _heuristical_preprocess_candidates_row, axis=1)


def _heuristical_preprocess_candidates_row(row):
    candidates_str = row["candidates"]
    candidates_list = json.loads(candidates_str)

    banned = {"bacteria", "virus"}

    result = set()
    for candidate in candidates_list:
        if candidate in banned:
            continue

        result.update(candidate.split(", "))
    result = list(result)

    return json.dumps(result)


def _heuristical_preprocess_labels(df):
    df["y_true_prep"] = df.apply(_heuristical_preprocess_labels_row, axis=1)


def _heuristical_preprocess_labels_row(row):
    result = set(row["y_true"].split(" or "))

    if "influzena" in result:
        result.remove("influzena")
        result.add("influenza")

    return json.dumps(result)


def _heuristical_predict(df):
    df["y_pred"] = df.apply(_heuristical_predict_row, axis=1)


def _heuristical_predict_row(row):
    candidates_str = row["candidates_prep"]
    candidates_set = set(json.loads(candidates_str))

    if not candidates_set:
        return "*not found"   # candidates_set is empty

    organism = get_one(candidates_set)
    if organism == "e coli":
        return "escherichia"

    level_1 = organism.split()[0]
    if level_1 == "hcv":
        return "hepatitis c virus"

    return level_1


def _heuristical_match(df):
    df["is_match"] = df.apply(_heuristical_match_row, axis=1)


def _heuristical_match_row(row):
    true_labels = json.loads(row["y_true_prep"])
    return any(
        true_label == row["y_pred"]
        for true_label in true_labels
    )


# ==============================================================================
# heuristical trial with Test Outcome information


def heuristical_with_test_outcome(df, save_to):
    df = df.copy()  # don't mutate the original DataFrame
    df.rename(columns={"level_1": "y_true"}, inplace=True)

    _heuristical_preprocess_candidates(df)
    _heuristical_preprocess_labels(df)

    _heuristical_with_test_outcome_predict(df)
    _heuristical_match(df)

    save_stats(df, save_to)


def _heuristical_with_test_outcome_predict(df):
    df["y_pred"] = df.apply(_heuristical_with_test_outcome_predict_row, axis=1)


def _heuristical_with_test_outcome_predict_row(row):
    if row["test_outcome"] == "negative":
        return "*not found"

    return _heuristical_predict_row(row)
