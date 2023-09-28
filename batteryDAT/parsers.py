# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""Parsers for loading data into pandas dataframes.

Contains functions which can be used to parse data from Biologic, Maccor,
and Basytec battery cyclers (which have been exported as .mpt, .txt, or .csv
files) into pandas dataframe objects.

"""

import numpy as np
import pandas as pd
from constants import CURRENT, DIS_CHARGE, NET_CHARGE, NS, TEMPERATURE, TIME, VOLTAGE

names_dictionary = {
    "time/s": TIME,
    "Ecell/V": VOLTAGE,
    "I/mA": CURRENT,
    "(Q-Qo)/mA.h": NET_CHARGE,
    "Temperature/ｰC": TEMPERATURE,
    "Q discharge/mA.h": DIS_CHARGE,
    "Ns changes": NS,
    "TestTime(s)": TIME,
    "Amps": CURRENT,
    "Volts": VOLTAGE,
    "Amp-hr": DIS_CHARGE,
    "Cyc#": "cycle",
    "Step": NS,
    "~Time[s]": TIME,
    "#Time[hh:mm:ss]": TIME,
    "I[A]": CURRENT,
    "U[V]": VOLTAGE,
    "Ah[Ah]": NET_CHARGE,
    "Ah-Cyc-Discharge": DIS_CHARGE,
    "Line": NS,
    "T1[°C]": TEMPERATURE,
}


def maccor(filename, skip_rows=2, use_columns=None):
    """Load Maccor data into a pandas dataframe.

    A function for loading Maccor data (in csv format) into a pandas
    dataframe and formatting columns etc into a standard format.

    filename: the name of the csv file to be parsed in
              (including the directory if not the current folder).
    skip_rows: the number of header rows in the csv file which aren't
                variable names/data. Default = 2 rows.
    use_columns: the variables/column titles to be imported. If values
                other than default are used, the new columns will not
                be changed by the renaming function.

    Returns a pandas DataFrame object containing the parsed data.
    """
    if use_columns is None:
        use_columns = ["TestTime(s)", "Amps", "Volts", "Amp-hr", "Cyc#", "Step"]
    data_df = pd.read_csv(filename, skiprows=skip_rows, usecols=use_columns)
    data_df = data_df.rename(columns=names_dictionary)
    # Change units of charge and create net charge column.
    data_df[DIS_CHARGE] = data_df[DIS_CHARGE] * 1000
    data_df[NET_CHARGE] = data_df[DIS_CHARGE] * np.sign(data_df[CURRENT])
    raw_dats = data_df.iloc[1:].copy()
    raw_dats.reset_index(inplace=True, drop=True)
    raw_dats.loc[raw_dats.shape[0]] = [0] * 7
    data_df["dQ"] = -(data_df[NET_CHARGE] - raw_dats[NET_CHARGE])
    # I can't remember why I made the next two rows...
    # some kind of noise in the data perhaps?
    data_df["dQ"][data_df["dQ"] < -1] = 0
    data_df["dQ"][data_df["dQ"] > 1] = 0
    data_df[NET_CHARGE] = np.cumsum(data_df["dQ"])
    return data_df


def basytec(filename, skip_rows=12, use_columns=None, separator=None):
    """Load Basytec data into a pandas dataframe.

    A function for loading Basytec data (in csv format) into a pandas
    dataframe and formatting columns etc into a standard format.

    filename: the name of the csv file to be parsed in
              (including the directory if not the current folder).
    skip_rows: the number of header rows in the csv file which aren't
                variable names/data. Default = 12 rows.
    use_columns: the variables/column titles to be imported. If values
                other than default are used, the new columns will not
                be changed by the renaming function.
    separator: the character used as a separator/delimiter in the csv file.
               Default is None, which uses default for pandas read_csv().

    Returns a pandas DataFrame object containing the parsed data.
    """
    if use_columns is None:
        use_columns = ["~Time[s]", "I[A]", "U[V]", "Ah[Ah]", "Ah-Cyc-Discharge", "Line"]
    data_df = pd.read_csv(
        filename,
        encoding="ansi",
        skiprows=skip_rows,
        usecols=use_columns,
        sep=separator,
    )
    # Convert time format to seconds passed.
    if "#Time[hh:mm:ss]" in use_columns:
        data_df["#Time[hh:mm:ss]"] = data_df["#Time[hh:mm:ss]"].str.replace(
            ",", "."
        ) / np.timedelta64(1, "s")
    if ("Ah-Cyc-Discharge" not in use_columns) and ("Ah-Step" in use_columns):
        data_df["Ah-Cyc-Discharge"] = data_df["Ah-Step"]
        data_df["Ah-Cyc-Discharge"].where(
            data_df["Ah-Cyc-Discharge"] < 0, 0, inplace=True
        )

    data_df = data_df.rename(columns=names_dictionary)
    # Change units of charge and current
    data_df[DIS_CHARGE] = data_df[DIS_CHARGE] * 1000
    data_df[NET_CHARGE] = data_df[NET_CHARGE] * 1000
    data_df[CURRENT] = data_df[CURRENT] * 1000
    return data_df


def biologic(filename, use_columns=None):
    """Load BT-Lab raw mpt files into Pandas dataframes.

    This uses pandas "read_csv" function and uses some default arguments
    for standard properties of the BT-Lab datafiles (encoding, etc.).

    Parameters
    ----------
    filename : str
        filename and directory of the mpt file to be parsed.
    use_columns : list, optional
        Column/variable names to import from the mpt file.
        The default is ["time/s", "Ecell/V", "I/mA", "(Q-Qo)/mA.h",
                        "Temperature/ｰC", "Q discharge/mA.h", "Ns changes"].

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing the specified data from the mpt file.

    """
    if use_columns is None:
        use_columns = [
            "time/s",
            "Ecell/V",
            "I/mA",
            "(Q-Qo)/mA.h",
            "Temperature/ｰC",
            "Q discharge/mA.h",
            "Ns changes",
        ]
    # Header sections are of variable length for .mpt files. The number
    # of header lines is specified in the header text (if it exists).
    sample = pd.read_csv(filename, sep=":", nrows=2, names=[1, 2], encoding="shift-jis")
    if sample.iloc[1, 0] == "Nb header lines ":
        skip_rows = int(sample.iloc[1, 1] - 1)
    else:
        skip_rows = 0
    data_df = pd.read_csv(
        filename,
        encoding="shift-jis",
        skiprows=skip_rows,
        sep="\t",
        usecols=use_columns,
    )
    data_df = data_df.rename(columns=names_dictionary)
    return data_df


def create_SOC(data_df, capacity=0.0):
    """Create a new column called `SOC (%)` from capacity column of dataframe.

    Uses the measured and nominal capacities to calculate the state of charge
    (SOC). Assumes that the data covers the full SoC range, and that the
    measured capacity is in mAh.

    Parameters
    ----------
    data_df : pandas dataframe
        Dataframe containing the battery timeseries data.
    capacity : float, optional
        The capacity to be used for SOC determination (in Ah). Usually this is
        the nominal capacity of the cell. Default value of 0 results in using
        the measured (maximum - minimum) values of net charge passed.

    Returns
    -------
    pandas series
        A pandas series containing the calculated `SOC (%)`.

    """
    if not capacity:
        capacity = (data_df[NET_CHARGE].max() - data_df[NET_CHARGE].min()) / 1000
    return 100 + (data_df[NET_CHARGE] - data_df[NET_CHARGE].max()) / (capacity * 10)


def index_time(data_df):
    """Subtract minimum value of the measured time to reset count to zero."""
    return data_df[TIME] - data_df[TIME].min()
