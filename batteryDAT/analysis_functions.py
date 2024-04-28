# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""analysis_functions module.

Contains various functions for analysing battery data, including:
    Incremental capacity and differential voltage analyses (ICA & DVA);
    OCV and ohmic resistance determination from pulse data;
    Degradation mode analysis (open-ciruit voltage fitting);
    Differential thermal voltammetry (DTV);
    Overvoltage and dynamic resistance analysis (DRA).

"""

import numpy as np
import pandas as pd

from batteryDAT.constants import (
    CURRENT,
    DIFF_CAPACITY,
    DIFF_RESISTANCE,
    DIS_CHARGE,
    DYN_RESISTANCE,
    NS,
    OCV,
    OHM_RESISTANCE,
    OVER_VOLTAGE,
    SOC,
    TEMPERATURE,
    TIME,
    TOTAL_RESISTANCE,
    VOLTAGE,
)


def r0_calc_dis(input_data):
    """Calculate of instantaneous resistance using ohms law.

    Calculate resistance from instantaneous voltage drop on application of
    a current. Can be used on any type of discharge test (CC or GITT),
    with the number of output values (rows in output df) equal to the number
    of 'pulses' (where CC is a single pulse). Alongside the 'R0' resistance,
    it also outputs the OCV and SoC values immediately prior to the
    pulse commencing.

    df: the input data should be a pandas dataframe object with columns of:
        ['Voltage (V)', 'Current (mA)', 'Ns changes', 'SOC (%)']
    output: a pandas dataframe object with columns of
        ['SOC (%)', 'R0 (Ohms)', 'OCV (V)'].

    """
    df = input_data.copy()
    ocv_vals = []
    r0_vals = []
    soc_vals = []
    # Biologic data contains 'flags' of where pulses begin (NS).
    pulse_indices = df[(df[NS] == 1) & (df[CURRENT] < 0)].index
    numPulses = len(pulse_indices)
    print(numPulses)
    for counter, row in enumerate(pulse_indices):
        # Problem with NS flag on some datasets causes a divide by zero error.
        if abs(df[CURRENT].iloc[row - 1] - df[CURRENT].iloc[row]) < abs(
            df[CURRENT].iloc[row - 2] - df[CURRENT].iloc[row - 1]
        ):
            row += -1
        ocv_vals.append(df[VOLTAGE].iloc[(row - 10) : (row - 1)].mean())
        r0_vals.append(
            (ocv_vals[counter] - df[VOLTAGE].iloc[row])
            / ((df[CURRENT].iloc[row - 1] - df[CURRENT].iloc[row]) / 1000)
        )
        soc_vals.append(df[SOC].iloc[row - 1].round(2))
    data = [[i, j, k] for i, j, k in zip(soc_vals, r0_vals, ocv_vals)]
    r0_output_dis = pd.DataFrame(data, columns=[SOC, OHM_RESISTANCE, OCV])
    return r0_output_dis


def resist_time(input_data, timestep=10):
    """Calculate of resistance at set time-interval using ohms law.

    Calculate resistance from the voltage drop a fixed time after the
    application of a current. Can be used on any type of discharge test
    (CC or GITT), with the number of output values (rows in output df) equal
    to the number of 'pulses' (where CC is a single pulse). Alongside the
    'R0' resistance, it also outputs the OCV and SoC values immediately prior
    to the pulse commencing.

    df: the input data should be a pandas dataframe object with columns of:
        ['Time (s)', Voltage (V)', 'Current (mA)', 'Ns changes', 'SOC (%)']
    timestep: float, default 10 seconds. The time interval for determining R.
    output: a pandas dataframe object with columns of
        ['SOC (%)', 'R0 (Ohms)', 'OCV (V)'].

    """
    df = input_data.copy()
    ocv_vals = []
    r0_vals = []
    soc_vals = []
    # Biologic data contains 'flags' of where pulses begin (NS).
    pulse_indices = df[(df[NS] == 1) & (df[CURRENT] < 0)].index
    numPulses = len(pulse_indices)
    print(numPulses)
    for counter, row in enumerate(pulse_indices):
        # Problem with NS flag on some datasets causes a divide by zero error.
        if abs(df[CURRENT].iloc[row - 1] - df[CURRENT].iloc[row]) < abs(
            df[CURRENT].iloc[row - 2] - df[CURRENT].iloc[row - 1]
        ):
            row += -1
        # Find the time at start of pulse and the sample interval.
        start_time = df[TIME].iloc[row - 1]
        time_delta = df[TIME].iloc[row + 1] - df[TIME].iloc[row]
        # Use these to find the index of the sample posistion.
        try:
            end_index = df[
                (df[TIME] > (start_time + timestep - time_delta))
                & (df[TIME] < (start_time + timestep + time_delta))
            ].index[0]
        except IndexError:
            print(f"Pulse {counter} terminated before timestep completed.")
            continue
        ocv_vals.append(df[VOLTAGE].iloc[(row - 10) : (row - 1)].mean())
        r0_vals.append(
            (ocv_vals[counter] - df[VOLTAGE].iloc[end_index])
            / ((df[CURRENT].iloc[row - 1] - df[CURRENT].iloc[end_index]) / 1000)
        )
        soc_vals.append(df[SOC].iloc[row - 1].round(2))
    data = [[i, j, k] for i, j, k in zip(soc_vals, r0_vals, ocv_vals)]
    r0_output_dis = pd.DataFrame(data, columns=[SOC, OHM_RESISTANCE, OCV])
    return r0_output_dis


def r0_calc_dis_maccor(input_data, numPulses):
    """Calculate of instantaneous resistance using ohms law.

    Same as 'r0_calc_dis', but modified to work with data from the Maccor.
    Requires the number of pulses to be input as an argument.
    See that function for help.

    """
    ocv_vals = []
    r0_vals = []
    soc_vals = []
    pulse_indices = []
    df = input_data.copy()
    for pulse in range(1, (numPulses + 1)):
        pulse_indices.append(df[df["cycle"] == (pulse)].index.min())
    for row, counter in enumerate(pulse_indices):
        ocv_vals.append(df[VOLTAGE].iloc[(row - 10) : (row - 1)].mean())
        r0_vals.append(
            (ocv_vals[counter] - df[VOLTAGE].iloc[row])
            / ((df[CURRENT].iloc[row - 1] - df[CURRENT].iloc[row]) / 1000)
        )
        soc_vals.append(df[SOC].iloc[row - 1].round(2))
    data = [[i, j, k] for i, j, k in zip(soc_vals, r0_vals, ocv_vals)]
    r0_output_dis = pd.DataFrame(data, columns=[SOC, OHM_RESISTANCE, OCV])
    return r0_output_dis


def r0_calc_dis_bastyec(input_data):
    """Calculate of instantaneous resistance using ohms law.

    Same as 'r0_calc_dis', but modified to work with data from the Basytec.
    Requires the number of pulses to be input as an argument.
    See that function for help.

    """
    og_df = input_data.copy()
    # Create column which highlights changes in NS
    og_df["NS_cha"] = og_df[NS].diff()
    # Find the start of the discharge section of the data and slice the df
    start_point = og_df[og_df[CURRENT] < -10].index.min()
    df = og_df.loc[start_point - 20 :, :]
    df.reset_index(inplace=True, drop=True)

    ocv_vals = []
    r0_vals = []
    soc_vals = []
    # Make use of our new column of NS changes.
    pulse_indices_on = df[
        (
            (df["NS_cha"] == -2.0)
            | (df["NS_cha"] == 3.0)
            | (df["NS_cha"] == 2.0) & (df[CURRENT] == 0)
        )
    ].index
    pulse_indices_off = df[(df["NS_cha"] == 1.0) & (df[CURRENT] == 0)].index
    pulse_indices_off = [
        pulse_indices_off[row] for row in range(len(pulse_indices_off)) if row % 2 == 0
    ]

    for on_index, off_index in zip(pulse_indices_on, pulse_indices_off):
        ocv_vals.append(df[VOLTAGE].iloc[(on_index - 10) : (on_index - 1)].mean())
        r0_vals.append(
            (df[VOLTAGE].iloc[off_index - 1] - df[VOLTAGE].iloc[off_index])
            / ((df[CURRENT].iloc[off_index - 1] - df[CURRENT].iloc[off_index]) / 1000)
        )
        soc_vals.append(df[SOC].iloc[on_index - 1].round(2))
    data = [[i, j, k] for i, j, k in zip(soc_vals, r0_vals, ocv_vals)]
    r0_output_dis = pd.DataFrame(data, columns=[SOC, OHM_RESISTANCE, OCV])
    return r0_output_dis


def overvoltage(input_data, input_GITT, data_type="CC", c_type="d"):
    """Determine the overvoltage using OCV and under-load data.

    A function which takes an input dataset to be analysed (any type of charge
    or discharge, signalled by the 'c_type' argument with values of 'c' or 'd')
    alongside a reference GITT dataset which contains OCV and R0 vs SOC data.
    The output is a pandas DF with 10 columns, including overvoltage, total
    resistance, dynamic resistance, etc.

    df_input: a pandas dataframe object containing the raw data.
    GITT_data: a pandas dataframe containing R0 data; such as output from the
        'r0_calc_dis' function.
    c_type: specifying whether it is charge ('c') or discharge ('d') data to
        be analysed.

    Output: a pandas dataframe object containing the processed data.

    """
    # Create copies of dataframes so originals are not affected.
    df_input = input_data.copy()
    GITT_data = input_GITT.copy()
    # Selecting only discharge data or only charge ***PROBLEMS IF USING GITT***
    if c_type == "c":
        df_input_discharge = df_input.loc[df_input[CURRENT] > 0, :].copy()
    else:
        df_input_discharge = df_input.loc[df_input[CURRENT] < 0, :].copy()

    # Create new df with linearly spaced SOC list, make SOC the index.
    df_ocv_r0 = pd.DataFrame(columns=[SOC, OHM_RESISTANCE, VOLTAGE], data=None)
    df_ocv_r0[SOC] = np.linspace(100, 0, 10001).round(2)
    df_ocv_r0.set_index(SOC, inplace=True)
    GITT_data.loc[:, SOC] = GITT_data.loc[:, SOC].round(2)
    GITT_data.set_index(SOC, inplace=True)
    df_input_discharge.loc[:, SOC] = df_input_discharge.loc[:, SOC].round(2)
    df_input_discharge.set_index(SOC, inplace=True)
    df_input_discharge = df_input_discharge.loc[
        ~df_input_discharge.index.duplicated(), :
    ]

    # Add values to interpolated df based on index (SOC), then interpolate.
    df_ocv_r0[OHM_RESISTANCE] = GITT_data[OHM_RESISTANCE]
    df_ocv_r0[VOLTAGE] = GITT_data[OCV]
    df_ocv_r0[OHM_RESISTANCE] = df_ocv_r0[OHM_RESISTANCE].astype("float64")
    df_ocv_r0[OHM_RESISTANCE].interpolate(method="polynomial", order=2, inplace=True)
    df_ocv_r0[VOLTAGE] = df_ocv_r0[VOLTAGE].astype("float64")
    df_ocv_r0[VOLTAGE].interpolate(method="polynomial", order=2, inplace=True)

    # Create the columns to go into the final dataframe
    time_out = df_input_discharge.loc[:, TIME]
    I_out = df_input_discharge.loc[:, CURRENT]
    Q_out = df_input_discharge.loc[:, DIS_CHARGE]
    T_out = df_input_discharge.loc[:, TEMPERATURE]
    V_out = df_input_discharge.loc[:, VOLTAGE]
    V_op_out = (
        df_ocv_r0[VOLTAGE] - df_input_discharge[VOLTAGE]
    )  # Contains lots of empty rows where index doesnt match up.
    V_op_out.dropna(
        inplace=True
    )  # Drops all the rows where the SOC index didnt match up.
    V_op_out.rename(OVER_VOLTAGE, inplace=True)
    OCV_out = V_out + V_op_out
    OCV_out.rename(OCV, inplace=True)
    r_eff_out = (
        V_op_out / -df_input_discharge[CURRENT] * 1000
    )  # Same as before, lots of empty rows from df_ocv_r0.
    r_eff_out.dropna(inplace=True)
    r_eff_out.rename(TOTAL_RESISTANCE, inplace=True)
    r_eff_corr_out = (
        r_eff_out - df_ocv_r0[OHM_RESISTANCE]
    )  # Same as before, lots of empty rows from df_ocv_r0.
    r_eff_corr_out.dropna(inplace=True)
    r_eff_corr_out.rename(DYN_RESISTANCE, inplace=True)
    r0_out = r_eff_out - r_eff_corr_out
    r0_out.rename(OHM_RESISTANCE, inplace=True)

    # Concatenate together the above series into a final dataframe.
    df_out = pd.concat(
        [
            time_out,
            I_out,
            Q_out,
            OCV_out,
            V_out,
            V_op_out,
            r0_out,
            r_eff_out,
            r_eff_corr_out,
            T_out,
        ],
        axis=1,
    )
    df_out.sort_values(TIME, inplace=True)
    df_out.reset_index(inplace=True)
    df_out[TIME] = df_out[TIME] - df_out[TIME].iloc[0]
    return df_out


def overvoltage_pulse(input_data, input_GITT, data_type="pulse", c_type="d"):
    """Determine overvoltage for a pulsed dataset.

    *HAS ISSUES FOR CC DATA - NEEDS EDITED/RENAMED TO BE ONLY FOR PULSED DATA*
    A function which takes an input dataset to be analysed (any type of charge
    or discharge, signalled by the 'c_type' argument with values of 'c' or 'd')
    alongside a reference GITT dataset which contains OCV and R0 vs SOC data.
    The output is a pandas DF with 10 columns, including overvoltage, total
    resistance, dynamic resistance, etc.

    df_input: a pandas dataframe object containing the raw data.
    GITT_data: a pandas dataframe containing R0 data; such as output from the
        'r0_calc_dis' function.
    c_type: specifying whether it is charge ('c') or discharge ('d') data to
        be analysed.

    Output: a pandas dataframe object containing the processed data.

    """
    # Create copies of dataframes so originals are not affected.
    df_input = input_data.copy()
    GITT_data = input_GITT.copy()
    # Selecting only discharge data or only charge ***PROBLEMS IF USING GITT***
    if c_type == "c":
        df_input_discharge = df_input.loc[df_input[CURRENT] > 0, :].copy()
    else:
        df_input_discharge = df_input.loc[df_input[CURRENT] < 0, :].copy()

    # Create new df with linearly spaced SOC list, make SOC the index.
    df_ocv_r0 = pd.DataFrame(columns=[SOC, OHM_RESISTANCE, VOLTAGE], data=None)
    df_ocv_r0[SOC] = np.linspace(100, 0, 10001).round(2)
    df_ocv_r0.set_index(SOC, inplace=True)
    GITT_data.loc[:, SOC] = GITT_data.loc[:, SOC].round(2)
    GITT_data.set_index(SOC, inplace=True)
    GITT_data = GITT_data.loc[~GITT_data.index.duplicated(), :]  # Check Flag.
    df_input_discharge.loc[:, SOC] = df_input_discharge.loc[:, SOC].round(2)
    df_input_discharge.set_index(SOC, inplace=True)
    df_input_discharge = df_input_discharge.loc[
        ~df_input_discharge.index.duplicated(), :
    ]

    # Add values to interpolated df based on index (SOC), then interpolate.
    df_ocv_r0[OHM_RESISTANCE] = GITT_data[OHM_RESISTANCE]
    df_ocv_r0[VOLTAGE] = GITT_data[OCV]
    df_ocv_r0[OHM_RESISTANCE] = df_ocv_r0[OHM_RESISTANCE].astype("float64")
    df_ocv_r0[OHM_RESISTANCE].interpolate(method="polynomial", order=2, inplace=True)
    df_ocv_r0[VOLTAGE] = df_ocv_r0[VOLTAGE].astype("float64")
    df_ocv_r0[VOLTAGE].interpolate(method="polynomial", order=2, inplace=True)

    df_input_discharge[OCV] = df_ocv_r0[VOLTAGE]
    df_input_discharge[OHM_RESISTANCE] = df_ocv_r0[OHM_RESISTANCE]
    df_input_discharge[OCV].interpolate(method="polynomial", order=2, inplace=True)
    df_input_discharge[OHM_RESISTANCE].interpolate(
        method="polynomial", order=2, inplace=True
    )
    df_input_discharge.set_index(TIME, inplace=True)
    df_input.set_index(TIME, inplace=True)
    df_input[OCV] = df_input_discharge[OCV]
    df_input[OHM_RESISTANCE] = df_input_discharge[OHM_RESISTANCE]
    df_input.reset_index(inplace=True)
    if c_type == "c":
        start_time = df_input[df_input[CURRENT] > 0][TIME].iloc[0]
    else:
        start_time = df_input[df_input[CURRENT] < 0][TIME].iloc[0]
    df_input = df_input[df_input[TIME] > start_time - 1]
    if (pd.isna(df_input[OCV].iloc[0])) & (df_input[CURRENT].iloc[0] == 0.0):
        df_input[OCV].iloc[0] = df_input[VOLTAGE].iloc[0]
    if (pd.isna(df_input[OHM_RESISTANCE].iloc[0])) & (df_input[CURRENT].iloc[0] == 0.0):
        df_input[OHM_RESISTANCE].iloc[0] = df_input[OHM_RESISTANCE].dropna().iloc[0]
    df_input[OCV].interpolate(method="pad", inplace=True)
    df_input[OHM_RESISTANCE].interpolate(method="pad", inplace=True)
    df_input[TIME] = df_input[TIME] - df_input[TIME].iloc[0]
    df_input[OVER_VOLTAGE] = df_input[VOLTAGE] - df_input[OCV]
    df_input[TOTAL_RESISTANCE] = (
        1000 * df_input[OVER_VOLTAGE] / df_input[CURRENT]
    )  # Divide by zero issue if no current
    df_input[DYN_RESISTANCE] = df_input[TOTAL_RESISTANCE] - df_input[OHM_RESISTANCE]
    df_input.replace(
        [np.inf, -np.inf], 0.0, inplace=True
    )  # Replace the infinite vals from division by zero to be zero instead.
    df_input.reset_index(inplace=True, drop=True)

    return df_input


def dQdV(input_data, dV_range=0.005, V_total=1.700, I_type="d"):
    """Incremental capacity analysis.

    A function for calculating dQ/dV data from a pandas DF containing "SOC",
    "CURRENT", and "VOLTAGE" columns. The function can calculate dQ/dV for
    discharge ('d'), charge ('c') or both ('b'), outputting either a single DF
    or a list of 2 DFs. Each DF contains "SOC", "VOLTAGE", and "dQ/dV" columns.
    The function uses a finite-difference method, with default dV values of
    5 mV with a total range of 1700 mV (i.e. 4.2 to 2.5 V).

    The function is more complicated due to the possibility of having
    discharge, charge, or both; opertions themselves are very simple though.

    """
    df_data = input_data.copy()

    if I_type in ["d", "b"]:
        df_dis = df_data[(df_data[CURRENT] < 0)].copy()
        Vmax_dis = df_dis[VOLTAGE].max()
        dQdV_dis = []
        V_dis = []
        soc_dis = []
    if I_type in ["c"]:
        df_cha = df_data[df_data[CURRENT] > 0].copy()
        Vmin_cha = df_cha[VOLTAGE].min()
        dQdV_cha = []
        V_cha = []
        soc_cha = []
    if I_type in ["b"]:
        df_cha = df_data[
            ((df_data[CURRENT] > 0) & (df_data[TIME] > df_dis[TIME].iloc[0]))
        ].copy()
        Vmin_cha = df_cha[VOLTAGE].min()
        dQdV_cha = []
        V_cha = []
        soc_cha = []

    # Determine the total number of 'segments' (datapoints in the output).
    Segments = int(V_total / dV_range)
    output_data = []

    for i in range(Segments):
        if I_type in ["d", "b"]:
            sub_df_dis = df_dis[
                (df_dis[VOLTAGE] < (Vmax_dis - dV_range * i))
                & (df_dis[VOLTAGE] > (Vmax_dis - dV_range * (i + 1)))
            ]
            Qcol_dis = sub_df_dis[SOC]
            Vcol_dis = sub_df_dis[VOLTAGE]

        if I_type in ["c", "b"]:
            sub_df_cha = df_cha[
                (
                    (df_cha[VOLTAGE] > (Vmin_cha + (dV_range * i)))
                    & (df_cha[VOLTAGE] < (Vmin_cha + (dV_range * (i + 1))))
                )
            ]
            Qcol_cha = sub_df_cha[SOC]
            Vcol_cha = sub_df_cha[VOLTAGE]

        if I_type in ["b"]:
            if len(Vcol_dis) < 1:
                if len(Vcol_cha) < 1:
                    continue
                dQ_cha = Qcol_cha.iloc[-1] - Qcol_cha.iloc[0]
                dV_cha = Vcol_cha.iloc[-1] - Vcol_cha.iloc[0]
                dQdV_cha.append(dQ_cha / dV_cha)
                V_cha.append(Vmin_cha + (dV_range * i))
                soc_cha.append(Qcol_cha.iloc[0])
                continue

        if I_type in ["d", "b"]:
            if len(Vcol_dis) < 1:
                continue
            dQ_dis = -(Qcol_dis.iloc[-1] - Qcol_dis.iloc[0])
            dV_dis = Vcol_dis.iloc[-1] - Vcol_dis.iloc[0]
            dQdV_dis.append(dQ_dis / dV_dis)
            V_dis.append(Vmax_dis - dV_range * i)
            soc_dis.append(Qcol_dis.iloc[0])

        if I_type in ["c", "b"]:
            if len(Vcol_cha) < 1:
                continue
            dQ_cha = Qcol_cha.iloc[-1] - Qcol_cha.iloc[0]
            dV_cha = Vcol_cha.iloc[-1] - Vcol_cha.iloc[0]
            dQdV_cha.append(dQ_cha / dV_cha)
            V_cha.append(Vmin_cha + (dV_range * i))
            soc_cha.append(Qcol_cha.iloc[0])

    if I_type in ["d", "b"]:
        dQdV_dis_data = [[k, l, m] for k, l, m in zip(soc_dis, V_dis, dQdV_dis)]
        dQdV_dis_data_output = pd.DataFrame(
            dQdV_dis_data, columns=[SOC, VOLTAGE, DIFF_CAPACITY]
        )
        output_data.append(dQdV_dis_data_output)

    if I_type in ["c", "b"]:
        dQdV_cha_data = [[k, l, m] for k, l, m in zip(soc_cha, V_cha, dQdV_cha)]
        dQdV_cha_data_output = pd.DataFrame(
            dQdV_cha_data, columns=[SOC, VOLTAGE, DIFF_CAPACITY]
        )
        output_data.append(dQdV_cha_data_output)

    if I_type in ["d", "c"]:
        return output_data[0]
    else:
        return output_data


def dRdQ_calc(input_data, dQ_range=1, Q_total=100):
    """Differential resistance analysis.

    A function for calculating the differential resistance (dR/dQ).
    This function requires an "Reff-R0 (Ohms)" column in the input DF (i.e.
    like that output from the 'overvoltage_new' function). The function uses a
    finite-difference method with default dQ (or dSOC) value of 1%. The output
    is a DF with 'SOC' and 'dR/dSOC (Ohms/%)' columns.

    """
    df_data = input_data.copy()
    Segments = int(Q_total / dQ_range)
    dRdQ = []
    soc = []

    # dRdQ = [[r[i]/q[i] for q, r in df_data] for i in  ]
    for i in range(Segments):
        sub_df = df_data[
            (df_data[SOC] < (Q_total - dQ_range * i))
            & (df_data[SOC] > (Q_total - dQ_range * (i + 1)))
        ]
        # sub_df.set_index((range(len(sub_df))), inplace=True)
        Qcol = sub_df[SOC]
        Rcol = sub_df[DYN_RESISTANCE]
        if len(Qcol) < 1:
            break
        dQ = -(Qcol.iloc[-1] - Qcol.iloc[0])
        dR = Rcol.iloc[-1] - Rcol.iloc[0]
        dRdQ.append(dR / dQ)
        soc.append(Q_total - dQ_range * i)

    data = [[soc_vals, dR_vals] for soc_vals, dR_vals in zip(soc, dRdQ)]
    data_output = pd.DataFrame(data, columns=[SOC, DIFF_RESISTANCE])
    return data_output
