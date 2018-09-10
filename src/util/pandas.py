import pandas as pd


def set_pandas_options():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", None)
