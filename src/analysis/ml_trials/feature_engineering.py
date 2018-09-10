from src.analysis.ml_trials.basic import baseline
from src.analysis.ml_trials.clustering import cluster_labelled
from src.analysis.ml_trials.splitting import split_by_organism
from src.modules.preprocessor import replace_organisms, replace_numbers, \
    remove_symbols, replace_hepatitis


def change_numbers(df, output, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    df["result_full_description"]\
        = df["result_full_description"].apply(replace_numbers)

    baseline(df, output, labels, save_to)


def change_organisms(df, output, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    def helper(row):
        return replace_organisms(
            row["result_full_description"], row["candidates"])

    df["result_full_description"] = df.apply(helper, axis=1)

    baseline(df, output, labels, save_to)


def change_all(df, output, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    def helper(row):
        return replace_organisms(
            replace_numbers(remove_symbols(row["result_full_description"])),
            row["candidates"]
        )

    df["result_full_description"] = df.apply(helper, axis=1)

    baseline(df, output, labels, save_to)


def change_all_but_organisms(df, output, labels, save_to):
    df = df.copy()  # don't mutate the original DataFrame

    df["result_full_description"] = df["result_full_description"].apply(
        lambda x: replace_numbers(remove_symbols(x))
    )

    baseline(df, output, labels, save_to)


def split_by_organism_better(df, labels, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    def helper(row):
        return replace_hepatitis(replace_organisms(
            replace_numbers(remove_symbols(row["result_full_description"])),
            row["candidates"]
        ))

    df["result_full_description"] = df.apply(helper, axis=1)

    split_by_organism(df, labels, save_to)


def change_numbers_organisms_kmeans(df, output, n_clusters, save_to):
    df = df.copy()   # don't mutate the original DataFrame

    def helper(row):
        return replace_organisms(
            replace_numbers(row["result_full_description"]), row["candidates"])

    df["result_full_description"] = df.apply(helper, axis=1)

    cluster_labelled(df, output, n_clusters, save_to)
