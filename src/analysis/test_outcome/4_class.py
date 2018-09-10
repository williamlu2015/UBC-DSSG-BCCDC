import os
from datetime import datetime

from root import from_root
from src.analysis.ml_trials.basic import data_size_trend, data_size, baseline
from src.analysis.ml_trials.clustering import cluster_labelled, \
    cluster_labelled_1000_features, cluster_labelled_1000_features_pca, \
    cluster_labelled_1000_features_lda, cluster_labelled_tfidf, \
    cluster_labelled_variance, cluster_all
from src.analysis.ml_trials.comparison import comparison_lr_rf_intersection, \
    comparison_rf_svm_intersection, comparison_lr_rf_difference, \
    comparison_all_intersection
from src.analysis.ml_trials.feature_engineering import change_organisms, \
    change_all, change_numbers, change_numbers_organisms_kmeans
from src.analysis.ml_trials.feature_selection import \
    feature_selection_variance_trend, feature_selection_chi2_trend, \
    feature_selection_variance, feature_selection_chi2
from src.analysis.ml_trials.parameter_tuning import penalty, \
    regularization_strength_lr, regularization_strength_svm, class_weight, \
    multi_class, sag, n_estimators_rf, degree_svm
from src.analysis.ml_trials.sampling import downsample_train, \
    downsample_train_test
from src.analysis.ml_trials.splitting import split_skewed, split_by_date_half, \
    split_by_date_quarter, split_by_dss_2_3, split_by_dss_3_2, \
    split_by_test_type_culture_train, split_by_test_type_culture_test, \
    split_by_test_type_antibody_nat, split_by_test_type_nat_antibody
from src.analysis.ml_trials.stop_words import stop_words_english, \
    stop_words_hepatitis, stop_words_min_df, stop_words_min_df_trend
from src.analysis.ml_trials.vectorizing import lemmatization, stemming, \
    character_ngrams, character_trigrams, ngram, tfidf
from src.analysis.ml_trials.visualizations import pca_2d, lda_2d, pca_3d, \
    lda_3d, t_sne
from src.modules.db import extract

OUTPUT = "test_outcome"
LABELS = ["positive", "negative", "indeterminate", "*missing"]
SAVE_TO = from_root("results\\test_outcome\\4_class")


def main():
    run_basic_trials()
    run_vectorizing_trials()
    run_stop_words_trials()
    run_feature_selection_trials()
    run_feature_engineering_trials()
    run_parameter_tuning_trials()
    run_splitting_trials()
    run_comparison_trials()
    run_sampling_trials()
    run_visualization_trials()
    run_clustering_trials()


def run_basic_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    baseline(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "baseline"))

    for size in [250, 500, 1000]:
        data_size(
            df, OUTPUT, LABELS, size,
            os.path.join(SAVE_TO, f"data_size_{size}"))

    data_size_trend(df, OUTPUT, os.path.join(SAVE_TO, "data_size_trend"))


def run_vectorizing_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

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
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    stop_words_english(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stop_words_english"))
    stop_words_hepatitis(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stop_words_hepatitis"))
    stop_words_min_df(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "stop_words_min_df"))
    stop_words_min_df_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "stop_words_min_df_trend"))


def run_feature_selection_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    feature_selection_variance(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "feature_selection_variance"))
    feature_selection_variance_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "feature_selection_variance_trend"))
    feature_selection_chi2(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "feature_selection_chi2"))
    feature_selection_chi2_trend(
        df, OUTPUT, os.path.join(SAVE_TO, "feature_selection_chi2_trend"))


def run_feature_engineering_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))
    change_numbers(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "change_numbers"))

    df = extract(from_root("sql\\test_outcome\\4_class_metamap.sql"))
    change_organisms(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "change_organisms"))
    change_all(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "change_all"))
    change_numbers_organisms_kmeans(
        df, OUTPUT, 4, os.path.join(SAVE_TO, "change_numbers_organisms_kmeans"))


def run_parameter_tuning_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

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
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))
    split_skewed(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_skewed"))

    df = extract(from_root("sql\\test_outcome\\4_class_date.sql"))
    split_by_date_half(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_date_half"))
    split_by_date_quarter(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_date_quarter"))

    df = extract(from_root("sql\\test_outcome\\4_class_dss.sql"))
    split_by_dss_2_3(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_dss_2_3"))
    split_by_dss_3_2(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "split_by_dss_3_2"))

    df = extract(from_root("sql\\test_outcome\\4_class_test_type.sql"))
    split_by_test_type_culture_train(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_culture_train"))
    split_by_test_type_culture_test(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_culture_test"))

    df = extract(from_root(
        "sql\\test_outcome\\4_class_test_type_antibody_nat.sql"))
    split_by_test_type_antibody_nat(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_antibody_nat"))
    split_by_test_type_nat_antibody(
        df, OUTPUT, LABELS,
        os.path.join(SAVE_TO, "split_by_test_type_nat_antibody"))


def run_comparison_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    comparison_lr_rf_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_lr_rf_intersection"))
    comparison_rf_svm_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_rf_svm_intersection"))
    comparison_lr_rf_difference(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_lr_rf_difference"))
    comparison_all_intersection(
        df, OUTPUT, os.path.join(SAVE_TO, "comparison_all_intersection"))


def run_sampling_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    downsample_train(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "downsample_train"))
    downsample_train_test(
        df, OUTPUT, LABELS, os.path.join(SAVE_TO, "downsample_train_test"))


def run_visualization_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    pca_2d(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "pca_2d"))
    lda_2d(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "lda_2d"))
    pca_3d(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "pca_3d"))
    lda_3d(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "lda_3d"))
    t_sne(df, OUTPUT, LABELS, os.path.join(SAVE_TO, "t_sne"))


def run_clustering_trials():
    df = extract(from_root("sql\\test_outcome\\4_class.sql"))

    cluster_labelled(df, OUTPUT, 4, os.path.join(SAVE_TO, "cluster_labelled"))
    cluster_labelled_1000_features(
        df, OUTPUT, 4, os.path.join(SAVE_TO, "cluster_labelled_1000_features"))
    cluster_labelled_1000_features_pca(
        df, OUTPUT, 4,
        os.path.join(SAVE_TO, "cluster_labelled_1000_features_pca"))
    cluster_labelled_1000_features_lda(
        df, OUTPUT, 4,
        os.path.join(SAVE_TO, "cluster_labelled_1000_features_lda"))
    cluster_labelled_tfidf(
        df, OUTPUT, 4, os.path.join(SAVE_TO, "cluster_labelled_tfidf"))
    cluster_labelled_variance(
        df, OUTPUT, 4, os.path.join(SAVE_TO, "cluster_labelled_variance"))

    df = extract(from_root("sql\\dataset.sql"))
    cluster_all(df, 4, os.path.join(SAVE_TO, "cluster_all"))


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    main()

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
