def split_by_index(df, is_test_index):
    df = df.copy()   # don't mutate the original DataFrame

    m, n = df.shape
    test_indices = [i for i in range(m) if is_test_index(i)]

    df_train = df.drop(df.index[test_indices])
    df_test = df.iloc[test_indices, :]
    return df_train, df_test


def downsample(df, output_name):
    df = df.copy()   # don't mutate the original DataFrame

    n = _min_class_num_rows(df, output_name)
    return df.groupby(output_name, as_index=False)\
        .apply(lambda x: x.sample(n))\
        .reset_index(drop=True)


def _min_class_num_rows(df, output):
    """
    Returns the number of rows in the class with the least number of rows.
    :param df: a data frame with all the rows
    :param output: the column of the data frame containing the class names
    :return: the number of rows in the class with the least number of rows
    """
    return df.groupby(output).size().min()
