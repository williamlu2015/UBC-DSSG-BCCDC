import pyodbc

import pandas as pd

from src.util.io import read_text


def extract(sql_filename, server="SDDBSBI002", database="DSSG"):
    sql = read_text(sql_filename)

    # "Trusted_connection=yes" tells SQL Server to use Windows Authentication
    db_string = "DRIVER={ODBC Driver 13 for SQL Server};" + \
                "Trusted_connection=yes;" + \
                "SERVER=" + server + ";" + \
                "DATABASE=" + database
    connection = pyodbc.connect(db_string)

    df = pd.read_sql(sql, connection)
    connection.close()

    return df
