import os
from datetime import datetime

from root import from_root
from src.analysis.ml_trials.basic import baseline, data_size, data_size_trend
from src.analysis.ml_trials.comparison import comparison_lr_rf_intersection, \
    comparison_rf_svm_intersection, comparison_lr_rf_difference, \
    comparison_all_intersection
from src.analysis.ml_trials.feature_engineering import change_numbers, \
    change_all_but_organisms
from src.analysis.ml_trials.feature_selection import feature_selection_variance, \
    feature_selection_variance_trend, feature_selection_chi2, \
    feature_selection_chi2_trend
from src.analysis.ml_trials.parameter_tuning import penalty, \
    regularization_strength_lr, regularization_strength_svm, class_weight, \
    multi_class, sag, n_estimators_rf, degree_svm
from src.analysis.ml_trials.splitting import split_skewed, split_by_date_half, \
    split_by_date_quarter, split_by_dss_2_3, split_by_dss_3_2, \
    split_by_test_type_culture_train, split_by_test_type_culture_test, \
    split_by_test_type_antibody_nat, split_by_test_type_nat_antibody
from src.analysis.ml_trials.stop_words import stop_words_english, \
    stop_words_min_df, stop_words_min_df_trend
from src.analysis.ml_trials.vectorizing import ngram, tfidf, stemming, \
    lemmatization, character_ngrams, character_trigrams
from src.modules.db import extract


OUTPUT = "level_1"
LABELS = None
SAVE_TO = from_root("results\\organism_name\\level_1_ml")


def main():
    run_basic_trials()
    run_vectorizing_trials()
    run_stop_words_trials()
    run_feature_selection_trials()
    run_feature_engineering_trials()
    run_parameter_tuning_trials()
    run_splitting_trials()
    run_comparison_trials()


def run_basic_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    baseline(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "baseline"))

    for size in [250, 500, 1000]:
        data_size(
            df, OUTPUT, LABELS, size,
            os.path.join(SAVE_TO, f"data_size_{size}"))

    data_size_trend(df, OUTPUT, os.path.join(SAVE_TO, "data_size_trend"))


def run_vectorizing_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    for ngram_range in [(1, 2), (1, 3), (2, 3)]:
        ngram(
            df, OUTPUT, LABELS, ngram_range,
            os.path.join(SAVE_TO, f"ngram_{ngram_range}"))

    tfidf(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "tfidf"))

    stemming(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stemming"))
    lemmatization(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "lemmatization"))
    character_ngrams(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "character_ngrams"))
    character_trigrams(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "character_trigrams"))


def run_stop_words_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    stop_words_english(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stop_words_english"))
    stop_words_min_df(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stop_words_min_df"))
    stop_words_min_df_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "stop_words_min_df_trend"))


def run_feature_selection_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    feature_selection_variance(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "feature_selection_variance"))
    feature_selection_variance_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "feature_selection_variance_trend"))
    feature_selection_chi2(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "feature_selection_chi2"))
    feature_selection_chi2_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "feature_selection_chi2_trend"))


def run_feature_engineering_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))
    change_numbers(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "change_numbers"))
    change_all_but_organisms(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "change_all_but_organisms"))


def run_parameter_tuning_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    penalty(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "penalty"))
    regularization_strength_lr(
        df, OUTPUT, os.path.join(SAVE_TO, "regularization_strength_lr"))
    regularization_strength_svm(
        df, OUTPUT, os.path.join(SAVE_TO, "regularization_strength_svm"))
    class_weight(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "class_weight"))
    multi_class(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "multi_class"))
    sag(df, OUTPUT, os.path.join(SAVE_TO, "sag"))
    n_estimators_rf(df, OUTPUT, os.path.join(SAVE_TO, "n_estimators_rf"))
    degree_svm(df, OUTPUT, os.path.join(SAVE_TO, "degree_svm"))


def run_splitting_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))
    split_skewed(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_skewed"))

    df = extract(from_root("sql\\organism_name\\level_1_ml_date.sql"))
    split_by_date_half(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_date_half"))
    split_by_date_quarter(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_date_quarter"))

    df = extract(from_root("sql\\organism_name\\level_1_ml_dss.sql"))
    split_by_dss_2_3(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_dss_2_3"))
    split_by_dss_3_2(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_dss_3_2"))

    df = extract(from_root("sql\\organism_name\\level_1_ml_test_type.sql"))
    split_by_test_type_culture_train(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_culture_train"))
    split_by_test_type_culture_test(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_culture_test"))

    df = extract(from_root(
        "sql\\organism_name\\level_1_ml_test_type_antibody_nat.sql"))
    split_by_test_type_antibody_nat(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_antibody_nat"))
    split_by_test_type_nat_antibody(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_nat_antibody"))


def run_comparison_trials():
    df = extract(from_root("sql\\organism_name\\level_1_ml.sql"))

    comparison_lr_rf_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_lr_rf_intersection"))
    comparison_rf_svm_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_rf_svm_intersection"))
    comparison_lr_rf_difference(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_lr_rf_difference"))
    comparison_all_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_all_intersection"))


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    main()

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
