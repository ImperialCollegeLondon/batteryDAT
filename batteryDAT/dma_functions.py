# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""Degradation mode analysis module."""

import numpy as np
import pandas as pd
from constants import CURRENT, DIS_CHARGE, OCV, SOC, VOLTAGE
from scipy import optimize

# --------- electrode-level calculations ---------


def composite_fit(
    component1_data, component2_data, composite_el_data, gr_cap_guess=0.85
):
    """Calculate the composition of a composite electrode."""
    # make a copy of the data for the composite electrode
    composite_el_data_edit = composite_el_data.copy()

    c1_data = component1_data.copy()
    c1_data["OCV"] = c1_data["OCV"].round(4)
    c1_data.drop_duplicates(
        subset=["OCV"], inplace=True
    )  # Duplicates cause issues when part of an index
    c1_data.set_index("OCV", inplace=True)
    c2_data = component2_data.copy()
    c2_data["OCV"] = c2_data["OCV"].round(4)
    c2_data.drop_duplicates(subset=["OCV"], inplace=True)
    c2_data.set_index("OCV", inplace=True)

    # Find the upper and lower voltage values used in the data fit.
    # Determined by the dataset with the lowest/highest values, respectively.
    if (
        composite_el_data_edit["OCV"].max() < component1_data["OCV"].max()
        and composite_el_data_edit["OCV"].max() < component2_data["OCV"].max()
    ):
        V_upper = composite_el_data_edit["OCV"].max()
    elif (
        component1_data["OCV"].max() < composite_el_data_edit["OCV"].max()
        and component1_data["OCV"].max() < component2_data["OCV"].max()
    ):
        V_upper = component1_data["OCV"].max()
    else:
        V_upper = component2_data["OCV"].max()

    if (
        composite_el_data_edit["OCV"].min() > component1_data["OCV"].min()
        and composite_el_data_edit["OCV"].min() > component2_data["OCV"].min()
    ):
        V_lower = composite_el_data_edit["OCV"].min()
    elif (
        component1_data["OCV"].min() > composite_el_data_edit["OCV"].min()
        and component1_data["OCV"].min() > component2_data["OCV"].min()
    ):
        V_lower = component1_data["OCV"].min()
    else:
        V_lower = component2_data["OCV"].min()

    # Set the voltage range used in the fitting.
    V_range = np.linspace(V_lower, V_upper, 10001)

    # V and SOC data from full_cell dataset need to be numpy arrays for fit.
    composite_el_data_edit = composite_el_data_edit[
        (
            (composite_el_data_edit["OCV"] < V_upper)
            & (composite_el_data_edit["OCV"] > V_lower)
        )
    ]
    el_V = np.array(composite_el_data_edit["OCV"])
    el_cap = np.array(composite_el_data_edit["z"])

    # Define the function which calculates a composite V vs Q curve based on
    # the two component curves (exact copy of 'calc_electrode_curve' function).
    # This function is passed into the scipy.optimize.curve_fit function below.
    def calc_electrode_cap(z_points, cap_comp1):
        cap_comp2 = 1.0 - cap_comp1

        # make new dataframes for the interpolation (component1 here)
        c1_data_int = pd.DataFrame(data=V_range, columns=["OCV"])
        # new DF with linearly spaced 'OCV' vals between V limits specified.
        c1_data_int["OCV"] = c1_data_int["OCV"].round(4)  # Remove rounding err
        c1_data_int.drop_duplicates(
            subset=["OCV"], inplace=True
        )  # Get rid of duplicates
        c1_data_int.set_index(
            "OCV", inplace=True
        )  # Make 'OCV' the index for matching 'z' against
        c1_data_int["z"] = c1_data["z"] * cap_comp1
        c1_data_int.interpolate(inplace=True)

        # same for component2
        c2_data_int = pd.DataFrame(
            data=V_range, columns=["OCV"]
        )  # Exactly the same as above, but for component2
        c2_data_int["OCV"] = c2_data_int["OCV"].round(4)
        c2_data_int.drop_duplicates(subset=["OCV"], inplace=True)
        c2_data_int.set_index("OCV", inplace=True)
        c2_data_int["z"] = c2_data["z"] * cap_comp2
        c2_data_int.interpolate(inplace=True)

        # calculate the composite electrode 'z' values from the two components
        cell_cap = pd.DataFrame(data=V_range, columns=["OCV"])
        cell_cap["OCV"] = cell_cap["OCV"].round(4)  # Get rid of rounding err
        cell_cap.drop_duplicates(subset=["OCV"], inplace=True)
        cell_cap.set_index("OCV", inplace=True)
        cell_cap["z"] = c1_data_int["z"] + c2_data_int["z"]
        cell_cap["z"] = cell_cap["z"].round(4)
        cell_cap.reset_index(inplace=True)
        cell_cap.drop_duplicates(subset=["z"], inplace=True)
        cell_cap.set_index(
            "z", inplace=True
        )  # Make the 'z' column the index for next section

        # The calculated composite electrode data is then made to match the
        # input (measured) composite electrode data in terms of 'z' (so that
        # they have the same number of datapoints for the optimisation).
        cell_out = pd.DataFrame(data=None, index=z_points.round(4))
        cell_out["OCV"] = cell_cap["OCV"]
        cell_out.interpolate(inplace=True)
        cell_out.reset_index(inplace=True)  # (not necessary - delete?)

        # Needs to be numpy array for optimisation function.
        output = np.array(cell_out["OCV"])

        print(cap_comp1, cap_comp2)

        # Give huge cost (i.e. error in the fit) if relative capacities don't
        # add up to 1 (they are meant ot be fractions of total)
        # if (cap_comp1 + cap_comp2) != 1.:
        #    output = np.ones((len(el_cap)))*10e5

        return output

    # Returns the fitted relative_caps and a covariance matrix
    z_out, z_cov = optimize.curve_fit(
        calc_electrode_cap,
        xdata=el_cap,
        ydata=el_V,
        p0=gr_cap_guess,
        bounds=(0.0, 1.0),
        diff_step=0.1,
    )

    return z_out, z_cov


def el_fit_error_check(NE_comp1_data, NE_comp2_data, electrode_data, fit_results):
    """Calculate the residual between real and calculated voltage curves."""
    el_data = electrode_data.copy()

    el_calc_data = calc_electrode_curve(
        NE_comp1_data, NE_comp2_data, el_data, *fit_results
    )

    el_data["OCV"] = el_data["OCV"].round(4)
    el_calc_data["OCV"] = el_calc_data["OCV"].round(4)
    print(el_calc_data.head())

    el_calc_data.drop_duplicates(subset=["OCV"], inplace=True)
    el_data.drop_duplicates(subset=["OCV"], inplace=True)

    if len(el_calc_data["OCV"]) < len(el_data["OCV"]):
        short_data = el_calc_data
        diff = pd.DataFrame(data=short_data["OCV"])
        diff.set_index("OCV", inplace=True)
        el_data.set_index("OCV", inplace=True)
        el_calc_data.set_index("OCV", inplace=True)

        diff["z error"] = el_data["z"] - el_calc_data["z"]

        err_array = np.array(diff["z error"])
        rmse_result = np.sqrt(np.square(err_array).mean())

    else:
        rmse_result = 10e5

    return rmse_result, diff


def el_fit_error_check_V(NE_comp1_data, NE_comp2_data, electrode_data, fit_results):
    """Calculate residual between calc & meas voltage curves."""
    el_data = electrode_data.copy()

    el_calc_data = calc_electrode_curve(
        NE_comp1_data, NE_comp2_data, el_data, *fit_results
    )

    el_data["z"] = el_data["z"].round(3)
    el_calc_data["z"] = el_calc_data["z"].round(3)
    print(el_calc_data.head())

    el_calc_data.drop_duplicates(subset=["z"], inplace=True)
    el_data.drop_duplicates(subset=["z"], inplace=True)

    if len(el_calc_data["z"]) < len(el_data["z"]):
        short_data = el_calc_data
        diff = pd.DataFrame(data=short_data["z"])
        diff = diff[diff["z"] > 0.04]
        diff.set_index("z", inplace=True)
        el_data.set_index("z", inplace=True)
        el_calc_data.set_index("z", inplace=True)

        diff["V error"] = el_data["OCV"] - el_calc_data["OCV"]

        err_array = np.array(diff["V error"])
        rmse_result = np.sqrt(np.square(err_array).mean())

    else:
        rmse_result = 10e5

    return rmse_result, diff


# --------- cell-level calculations ---------


def format_el_component_data(component1_data, component2_data):
    """Format data for two components of a composite half cell.

    Replaced by "format_el_component_data1" function. The difference between
    the two is whether the voltage limits are determined by the lowest or
    highest high and low voltage. This version takes the highest high and
    lowest low, with extrapolation for the limiting dataset.

    """
    c1_data = component1_data.copy()
    c2_data = component2_data.copy()

    # Add points to the end of the datasets with higher/lower lower/upper
    # voltage limits, so that the 'calc_electrode_curve' function can operate
    # over the full voltage range.
    # Also, use those limits to make a linearly-spaced voltage series between
    # those limits called 'V_ranges'
    if component1_data["OCV"].max() < component2_data["OCV"].max():
        V_upper = component2_data["OCV"].max()
        c1_data.loc[0, "OCV"] = V_upper
    else:
        V_upper = component1_data["OCV"].max()
        c2_data.loc[0, "OCV"] = V_upper

    if component1_data["OCV"].min() > component2_data["OCV"].min():
        V_lower = component2_data["OCV"].min()
        c1_data.loc[c1_data.index.max(), "OCV"] = V_lower
    else:
        V_lower = component1_data["OCV"].min()
        c2_data.loc[c2_data.index.max(), "OCV"] = V_lower

    V_ranges = np.unique(np.linspace(V_lower, V_upper, 10001).round(decimals=4))

    c1_data["OCV"] = c1_data["OCV"].round(4)
    c1_data.drop_duplicates(
        subset=["OCV"], inplace=True
    )  # Duplicates cause issues when part of an index
    c1_data.set_index("OCV", inplace=True)

    c2_data["OCV"] = c2_data["OCV"].round(4)
    c2_data.drop_duplicates(subset=["OCV"], inplace=True)
    c2_data.set_index("OCV", inplace=True)

    return c1_data, c2_data, V_ranges


def format_el_component_data1(component1_data, component2_data):
    """Format data for two components of a composite half cell.

    This replaces the "format_el_component_data" function. The difference
    between the two is whether the voltage limits are determined by the lowest
    or highest high and low voltage. This version takes the lowest high and
    highest low, thereby not requiring extrapolation.

    """
    c1_data = component1_data.copy()
    c2_data = component2_data.copy()

    if component1_data[OCV].max() < component2_data[OCV].max():
        V_upper = component1_data[OCV].max()
    else:
        V_upper = component2_data[OCV].max()

    if component1_data[OCV].min() > component2_data[OCV].min():
        V_lower = component1_data[OCV].min()
    else:
        V_lower = component2_data[OCV].min()

    c1_data = c1_data[((c1_data[OCV] <= V_upper) & (c1_data[OCV] >= V_lower))]
    c2_data = c2_data[((c2_data[OCV] <= V_upper) & (c2_data[OCV] >= V_lower))]

    V_ranges = np.unique(np.linspace(V_lower, V_upper, 10001).round(decimals=4))

    c1_data["z"] = (c1_data["z"] - c1_data["z"].min()) / (
        c1_data["z"].max() - c1_data["z"].min()
    )
    c1_data[OCV] = c1_data[OCV].round(4)
    c1_data.drop_duplicates(
        subset=[OCV], inplace=True
    )  # Duplicates cause issues when part of an index
    c1_data.set_index(OCV, inplace=True)

    c2_data["z"] = (c2_data["z"] - c2_data["z"].min()) / (
        c2_data["z"].max() - c2_data["z"].min()
    )
    c2_data[OCV] = c2_data[OCV].round(4)
    c2_data.drop_duplicates(subset=[OCV], inplace=True)
    c2_data.set_index(OCV, inplace=True)

    return c1_data, c2_data, V_ranges


def calc_electrode_curve(component1_data, component2_data, cap_comp1, V_range):
    """Simulate a V curve for a composite electrode."""
    cap_comp2 = 1.0 - cap_comp1

    # make new dataframes for the interpolation (component1 here)
    c1_data_int = pd.DataFrame(data=V_range, columns=[OCV])
    c1_data_int.set_index(OCV, inplace=True)
    c1_data_int["z"] = component1_data["z"] * cap_comp1
    c1_data_int.interpolate(inplace=True)

    # same for component2
    c2_data_int = pd.DataFrame(
        data=V_range, columns=[OCV]
    )  # Exactly the same as above, but for component2
    c2_data_int.set_index(OCV, inplace=True)
    c2_data_int["z"] = component2_data["z"] * cap_comp2
    c2_data_int.interpolate(inplace=True)

    # calculate the composite electrode 'z' values from the two components
    electrode_curve = pd.DataFrame(data=V_range, columns=[OCV])
    electrode_curve.set_index(OCV, inplace=True)
    electrode_curve["z"] = c1_data_int["z"] + c2_data_int["z"]
    electrode_curve["z"] = electrode_curve["z"].round(5)
    electrode_curve.reset_index(inplace=True)
    electrode_curve.drop_duplicates(subset=["z"], inplace=True)
    electrode_curve.set_index("z", inplace=True)

    return electrode_curve


def check_electrode_data(input_data, electrode_type=None):
    """Check electrode data is compatible format."""
    if input_data is None:
        input_data = str(input(f"Filepath for {electrode_type} data:"))
    if isinstance(input_data, str):
        try:
            electrode_data = pd.read_csv(input_data)
        except FileNotFoundError:
            print(f"Cannot find valid {electrode_type} data (.csv file)")
            return
    elif isinstance(input_data, pd.core.frame.DataFrame):
        electrode_data = input_data

    if all([names in electrode_data.columns for names in [OCV, "z"]]):
        return electrode_data
    elif any([names in electrode_data.columns for names in [VOLTAGE, "OCV", "V"]]):
        electrode_data.rename(
            columns={VOLTAGE: OCV, "OCV": OCV, "V": OCV}, inplace=True
        )
    else:
        OCV_name = str(input('Name of the "OCV" column in {electrode_type} data:'))
        z_name = str(input('Name of the "z" column in {electrode_type} data:'))
        if OCV_name and z_name in electrode_data.columns.values:
            electrode_data.rename(columns={OCV_name: OCV, z_name: "z"}, inplace=True)
        else:
            print(f"Columns could not be found in the {electrode_type} data.")
            return
    return electrode_data


def stoich_OCV_fit_multi_comp(
    anode_comp1_data,
    anode_comp2_data,
    cathode_data,
    full_cell,
    z_guess=None,
    diff_step_size=None,
):
    """Find electrode-level SoCs from cell-level pOCV data."""
    if not z_guess:
        z_guess = [0.1, 0.002, 0.95, 0.85, 0.84]
    if not diff_step_size:
        diff_step_size = 0.01
    # Format the negative electrode data (both components)
    ano_c1_data, ano_c2_data, V_range = format_el_component_data1(
        anode_comp1_data, anode_comp2_data
    )

    cell_V = np.array(full_cell[VOLTAGE])
    cell_SOC = np.array(full_cell[SOC])

    pe_data = cathode_data.copy()
    pe_data["z"] = pe_data["z"].round(5)
    pe_data.set_index("z", inplace=True)
    pe_data = pe_data.loc[~pe_data.index.duplicated(), :]

    # Define the function which calculates a full cell OCV based on the 1/2
    # cell data (exact copy of 'calc_full_cell_OCV_standalone').
    # This function is passed into the scipy.optimize.curve_fit function below.
    # As well as the defined arguments, it requires 1/2 cell data formatted
    # above, 'SOC_points' taken from full_cell dataset, and 'z_pe_lo' etc taken
    # from z_guess.
    def calc_full_cell_OCV_multi(
        SOC_points, z_pe_lo, z_ne_lo, z_pe_hi, z_ne_hi, comp1_frac
    ):
        ne_data = calc_electrode_curve(ano_c1_data, ano_c2_data, comp1_frac, V_range)

        z_ne = np.unique(np.linspace(z_ne_lo, z_ne_hi, 10001).round(decimals=5))
        z_pe = np.unique(np.linspace(z_pe_lo, z_pe_hi, 10001).round(decimals=5))

        ne_data_int = pd.DataFrame(data=z_ne, columns=["z"])
        ne_data_int.set_index("z", inplace=True)
        ne_data_int[OCV] = ne_data[OCV]
        ne_data_int.interpolate(inplace=True)
        ne_data_int.reset_index(inplace=True)

        # same for pos electrode
        pe_data_int = pd.DataFrame(data=z_pe, columns=["z"])
        pe_data_int.set_index("z", inplace=True)
        pe_data_int[OCV] = pe_data[OCV]
        pe_data_int.interpolate(inplace=True)
        pe_data_int.reset_index(inplace=True)

        # Calculate full cell ocv from 1/2 cell datasets.
        # Potential difference between PE and NE based on indices (NOT 'z').
        cell_ocv = pd.DataFrame(data=None)
        cell_ocv[OCV] = pe_data_int[OCV] - ne_data_int[OCV]
        cell_ocv[SOC] = np.linspace(0, 1, len(cell_ocv[OCV]))
        cell_ocv.set_index(SOC, inplace=True)

        cell_out = pd.DataFrame(data=None, index=SOC_points.round(5))
        cell_out[OCV] = cell_ocv[OCV]
        cell_out.interpolate(inplace=True)

        output = np.array(cell_out[OCV])

        return output

    # Optimisation returns fitted z_values and a covariance matrix.
    z_out, z_cov = optimize.curve_fit(
        calc_full_cell_OCV_multi,
        xdata=cell_SOC,
        ydata=cell_V,
        p0=z_guess,
        bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        diff_step=diff_step_size,
    )

    # Calculate capacities and offset of the full and 1/2 cells.
    PE_lo, NE_lo, PE_hi, NE_hi, comp1_frac = z_out
    cell_capacity = full_cell[DIS_CHARGE].max()
    ano_tot_cap = cell_capacity / (NE_hi - NE_lo)
    ano_comp1_cap = ano_tot_cap * comp1_frac
    ano_comp2_cap = ano_tot_cap * (1.0 - comp1_frac)
    cat_cap = cell_capacity / (PE_hi - PE_lo)
    # Offset is (lower bound of PE * PE cap) minus (lower bound of NE * NE cap)
    offset = (z_out[0] * cat_cap) - (z_out[1] * ano_tot_cap)
    stoic_params = [
        cell_capacity,
        cat_cap,
        ano_tot_cap,
        ano_comp1_cap,
        ano_comp2_cap,
        offset,
    ]

    return (z_out, z_cov, stoic_params)


def simulate_composite_OCV(
    anode_comp1_data,
    anode_comp2_data,
    cathode_data,
    SOC_points,
    z_pe_lo,
    z_ne_lo,
    z_pe_hi,
    z_ne_hi,
    comp1_frac,
):
    """Simulate a cell-level pOCV from 1/2 cell data."""
    ano_c1_data, ano_c2_data, V_range = format_el_component_data1(
        anode_comp1_data, anode_comp2_data
    )

    pe_data = cathode_data.copy()
    pe_data["z"] = pe_data["z"].round(5)
    pe_data.set_index("z", inplace=True)
    pe_data = pe_data.loc[~pe_data.index.duplicated(), :]

    ne_data = calc_electrode_curve(ano_c1_data, ano_c2_data, comp1_frac, V_range)

    z_ne = np.unique(np.linspace(z_ne_lo, z_ne_hi, 10001).round(decimals=5))
    z_pe = np.unique(np.linspace(z_pe_lo, z_pe_hi, 10001).round(decimals=5))

    # make new dataframes for the interpolation
    ne_data_int = pd.DataFrame(data=z_ne, columns=["z"])
    ne_data_int.set_index("z", inplace=True)
    ne_data_int["OCV"] = ne_data["OCV"]
    ne_data_int.interpolate(inplace=True)
    ne_data_int.reset_index(inplace=True)

    # same for pos electrode
    pe_data_int = pd.DataFrame(data=z_pe, columns=["z"])
    pe_data_int.set_index("z", inplace=True)
    pe_data_int["OCV"] = pe_data["OCV"]
    pe_data_int.interpolate(inplace=True)
    pe_data_int.reset_index(inplace=True)

    # calculate full cell ocv from 1/2 cell datasets
    cell_ocv = pd.DataFrame(data=None)
    cell_ocv["OCV"] = pe_data_int["OCV"] - ne_data_int["OCV"]
    cell_ocv[SOC] = np.linspace(0, 1, len(cell_ocv["OCV"]))
    cell_ocv.set_index(SOC, inplace=True)

    # match calculated full cell data with input (measured) full cell data
    # in terms of SOC (for same number of datapoints for the optimisation)
    cell_out = pd.DataFrame(data=None, index=SOC_points.round(5))
    cell_out["OCV"] = cell_ocv["OCV"]
    cell_out.interpolate(inplace=True)
    cell_out.reset_index(inplace=True)

    return cell_out, ne_data_int, ne_data, pe_data_int


def DM_calc_multi_comp(neg_el_comp1, neg_el_comp2, pos_el, BoL_cell, aged_cells):
    """Degradation mode analysis for cells with composite electrodes."""
    # Format the BoL full cell data
    BoL_cell_data = BoL_cell[BoL_cell[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
    BoL_cell_data.reset_index(inplace=True, drop=True)
    BoL_cell_data[SOC] = 1 - (
        BoL_cell_data[DIS_CHARGE] / BoL_cell_data[DIS_CHARGE].max()
    )

    # Perform the optimisation fit for the BoL data
    z_BoL, z_BoL_cov, BoL_params = stoich_OCV_fit_multi_comp(
        anode_comp1_data=neg_el_comp1,
        anode_comp2_data=neg_el_comp2,
        cathode_data=pos_el,
        full_cell=BoL_cell_data,
    )

    # Make a list of z_values and cell_parameters and populate with BoL data
    z_list = [z_BoL]
    param_list = [BoL_params]
    cov_matrix = [np.sqrt(np.diag(z_BoL_cov))]

    counter_val = 0
    # Perform fit for each of the aged cell datasets (using z_BoL as z_guess)
    for aged_data in aged_cells:
        # Format the aged full cell data
        aged_cell_data = aged_data[aged_data[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
        aged_cell_data.reset_index(inplace=True, drop=True)
        aged_cell_data[SOC] = 1 - (
            aged_cell_data[DIS_CHARGE] / aged_cell_data[DIS_CHARGE].max()
        )

        # Perform the optimisation fit for the aged full cell data
        z_EoL, z_EoL_cov, aged_params = stoich_OCV_fit_multi_comp(
            anode_comp1_data=neg_el_comp1,
            anode_comp2_data=neg_el_comp2,
            cathode_data=pos_el,
            full_cell=aged_cell_data,
            z_guess=z_list[counter_val],
        )
        z_list.append(z_EoL)
        param_list.append(aged_params)
        cov_matrix.append(np.sqrt(np.diag(z_EoL_cov)))
        counter_val += 1

    # Make dataframes using the lists complied above
    # z_parameter_df = pd.DataFrame(data=z_list,
    #                              columns=['PE_lo',
    #                                       'NE_lo',
    #                                       'PE_hi',
    #                                       'NE_hi',
    #                                       'Gr_frac'
    #                                       ]
    #                              )
    sto_param_df = pd.DataFrame(
        data=param_list,
        columns=[
            "Cell Capacity",
            "PE Capacity",
            "NE(tot) Capacity",
            "NE(Gr) Capacity",
            "NE(Si) Capacity",
            "Offset",
        ],
    )

    # Calculate DM parameters from the stoic_parameter dataframe above
    SoH = sto_param_df["Cell Capacity"] / sto_param_df["Cell Capacity"][0]
    LAM_pe = 1 - (sto_param_df["PE Capacity"] / sto_param_df["PE Capacity"][0])
    LAM_ne_tot = 1 - (
        sto_param_df["NE(tot) Capacity"] / sto_param_df["NE(tot) Capacity"][0]
    )
    LAM_ne_Gr = 1 - (
        sto_param_df["NE(Gr) Capacity"] / sto_param_df["NE(Gr) Capacity"][0]
    )
    LAM_ne_Si = 1 - (
        sto_param_df["NE(Si) Capacity"] / sto_param_df["NE(Si) Capacity"][0]
    )
    LLI = (
        sto_param_df["PE Capacity"][0]
        - sto_param_df["PE Capacity"]
        - (sto_param_df["Offset"][0] - sto_param_df["Offset"])
    ) / sto_param_df["Cell Capacity"][0]

    # Compile the DM parameters into a dataframe
    DM_df = pd.DataFrame(
        data={
            "SoH": SoH,
            "LAM PE": LAM_pe,
            "LAM NE_tot": LAM_ne_tot,
            "LAM NE_Gr": LAM_ne_Gr,
            "LAM NE_Si": LAM_ne_Si,
            "LLI": LLI,
        }
    )

    return DM_df, sto_param_df


def DM_error_check(NE_comp1_data, NE_comp2_data, PE_data, cell_data, fit_results):
    """Check RMSE of fitted pOCV curve against measured one."""
    cell_calc_data, _, _, _ = simulate_composite_OCV(
        NE_comp1_data, NE_comp2_data, PE_data, cell_data[SOC], *fit_results
    )

    if len(cell_calc_data["OCV"]) == len(cell_data[VOLTAGE]):
        diff = pd.DataFrame(data=cell_data[SOC])
        diff["V error"] = cell_data[VOLTAGE] - cell_calc_data["OCV"]
        err_array = np.array(diff["V error"])
        rmse_result = np.sqrt(np.square(err_array).mean())
    else:
        rmse_result = 10e5

    return rmse_result


def DM_calc_multi_comp_long(
    neg_el_comp1, neg_el_comp2, pos_el, BoL_cell, aged_cells, carry_guess=True
):
    """Degradation mode analysis for a cell at multiple SoH's."""
    # Format the BoL full cell data
    BoL_cell_data = BoL_cell[BoL_cell[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
    BoL_cell_data.reset_index(inplace=True, drop=True)
    BoL_cell_data[SOC] = 1 - (
        BoL_cell_data[DIS_CHARGE] / BoL_cell_data[DIS_CHARGE].max()
    )

    # Perform the optimisation fit for the BoL data
    z_BoL, z_BoL_cov, BoL_params = stoich_OCV_fit_multi_comp(
        anode_comp1_data=neg_el_comp1,
        anode_comp2_data=neg_el_comp2,
        cathode_data=pos_el,
        full_cell=BoL_cell_data,
    )

    # Calculate error of the fit
    err_BoL = DM_error_check(neg_el_comp1, neg_el_comp2, pos_el, BoL_cell_data, z_BoL)

    # Make a list of z_values and cell_parameters and populate with BoL data
    z_list = [z_BoL]
    param_list = [BoL_params]
    # cov_matrix = [np.sqrt(np.diag(z_BoL_cov))]
    err_list = [err_BoL]

    counter_val = 0

    for aged_data in aged_cells:
        # Format the aged full cell data
        aged_cell_data = aged_data[aged_data[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
        aged_cell_data.reset_index(inplace=True, drop=True)
        aged_cell_data[SOC] = 1 - (
            aged_cell_data[DIS_CHARGE] / aged_cell_data[DIS_CHARGE].max()
        )

        if not carry_guess:
            guess_values = [0.1, 0.002, 0.95, 0.85, 0.84]
        else:
            guess_values = z_list[counter_val]

        # Perform the optimisation fit for the aged full cell data
        z_EoL, z_EoL_cov, aged_params = stoich_OCV_fit_multi_comp(
            anode_comp1_data=neg_el_comp1,
            anode_comp2_data=neg_el_comp2,
            cathode_data=pos_el,
            full_cell=aged_cell_data,
            z_guess=guess_values,
        )

        # Calculate error of fit (RMSE of fitted curve minus actual data)
        err_EoL = DM_error_check(
            neg_el_comp1, neg_el_comp2, pos_el, aged_cell_data, z_EoL
        )

        iter_val = 0
        while (
            aged_params[4] > param_list[counter_val][4]
            or aged_params[2] > param_list[counter_val][2]
        ) and iter_val < 10:
            iter_val += 1
            z_EoL, z_EoL_cov, aged_params = stoich_OCV_fit_multi_comp(
                anode_comp1_data=neg_el_comp1,
                anode_comp2_data=neg_el_comp2,
                cathode_data=pos_el,
                full_cell=aged_cell_data,
                z_guess=z_list[counter_val],
                diff_step_size=0.1,
            )
            # Calculate error of fit (RMSE of fitted curve minus actual data)
            err_EoL = DM_error_check(
                neg_el_comp1, neg_el_comp2, pos_el, aged_cell_data, z_EoL
            )

        z_list.append(z_EoL)
        param_list.append(aged_params)
        # cov_matrix.append(np.sqrt(np.diag(z_EoL_cov)))
        err_list.append(err_EoL)
        counter_val += 1

    # Make dataframes using the lists complied above
    z_parameter_df = pd.DataFrame(
        data=z_list, columns=["PE_lo", "NE_lo", "PE_hi", "NE_hi", "Gr_frac"]
    )
    sto_param_df = pd.DataFrame(
        data=param_list,
        columns=[
            "Cell Capacity",
            "PE Capacity",
            "NE(tot) Capacity",
            "NE(Gr) Capacity",
            "NE(Si) Capacity",
            "Offset",
        ],
    )
    err_df = pd.DataFrame(data={"RMSE (V)": err_list})

    # Calculate DM parameters from the stoic_parameter dataframe above
    SoH = sto_param_df["Cell Capacity"] / sto_param_df["Cell Capacity"][0]
    LAM_pe = 1 - (sto_param_df["PE Capacity"] / sto_param_df["PE Capacity"][0])
    LAM_ne_tot = 1 - (
        sto_param_df["NE(tot) Capacity"] / sto_param_df["NE(tot) Capacity"][0]
    )
    LAM_ne_Gr = 1 - (
        sto_param_df["NE(Gr) Capacity"] / sto_param_df["NE(Gr) Capacity"][0]
    )
    LAM_ne_Si = 1 - (
        sto_param_df["NE(Si) Capacity"] / sto_param_df["NE(Si) Capacity"][0]
    )
    LLI = (
        sto_param_df["PE Capacity"][0]
        - sto_param_df["PE Capacity"]
        - (sto_param_df["Offset"][0] - sto_param_df["Offset"])
    ) / sto_param_df["Cell Capacity"][0]

    # Compile the DM parameters into a dataframe
    DM_df = pd.DataFrame(
        data={
            "SoH": SoH,
            "LAM PE": LAM_pe,
            "LAM NE_tot": LAM_ne_tot,
            "LAM NE_Gr": LAM_ne_Gr,
            "LAM NE_Si": LAM_ne_Si,
            "LLI": LLI,
        }
    )

    return DM_df, sto_param_df, err_df, z_parameter_df


def stoich_OCV_fit(
    anode_data, cathode_data, full_cell, z_guess=None, diff_step_size=None
):
    """Determine electrode-level SoCs from cell-level pOCV data."""
    if not z_guess:
        z_guess = [0.1, 0.002, 0.95, 0.85]
    if not diff_step_size:
        diff_step_size = 0.01

    cell_V = np.array(full_cell[VOLTAGE])
    cell_SOC = np.array(full_cell[SOC])

    pe_data = cathode_data.copy()
    pe_data["z"] = pe_data["z"].round(5)
    pe_data.set_index("z", inplace=True)
    pe_data = pe_data.loc[~pe_data.index.duplicated(), :]
    ne_data = anode_data.copy()
    ne_data["z"] = ne_data["z"].round(5)
    ne_data.set_index("z", inplace=True)
    ne_data = ne_data.loc[~ne_data.index.duplicated(), :]

    def calc_full_cell_OCV(SOC_points, z_pe_lo, z_ne_lo, z_pe_hi, z_ne_hi):
        z_ne = np.linspace(z_ne_lo, z_ne_hi, 100001)
        z_pe = np.linspace(z_pe_lo, z_pe_hi, 100001)

        # make new dataframes for the interpolation
        ne_data_int = pd.DataFrame(data=z_ne, columns=["z"])
        ne_data_int["z"] = ne_data_int["z"].round(5)
        ne_data_int.set_index("z", inplace=True)
        ne_data_int[OCV] = ne_data[OCV]
        ne_data_int.interpolate(inplace=True)
        ne_data_int.reset_index(inplace=True)

        # same for pos electrode
        pe_data_int = pd.DataFrame(data=z_pe, columns=["z"])
        pe_data_int["z"] = pe_data_int["z"].round(5)
        pe_data_int.set_index("z", inplace=True)
        pe_data_int[OCV] = pe_data[OCV]
        pe_data_int.interpolate(inplace=True)
        pe_data_int.reset_index(inplace=True)

        # calculate full cell ocv from 1/2 cell datasets
        cell_ocv = pd.DataFrame(data=None)
        cell_ocv[OCV] = pe_data_int[OCV] - ne_data_int[OCV]
        cell_ocv[SOC] = np.linspace(0, 1, len(cell_ocv[OCV]))
        cell_ocv.set_index(SOC, inplace=True)

        cell_out = pd.DataFrame(data=None, index=SOC_points.round(5))
        cell_out[OCV] = cell_ocv[OCV]
        cell_out.interpolate(inplace=True)
        cell_out.reset_index(inplace=True)

        # Optimisation function requires numpy arrays for data.
        output = np.array(cell_out[OCV])
        return output

    z_out, z_cov = optimize.curve_fit(
        calc_full_cell_OCV,
        xdata=cell_SOC,
        ydata=cell_V,
        p0=z_guess,
        bounds=([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        diff_step=diff_step_size,
    )

    PE_lo, NE_lo, PE_hi, NE_hi = z_out
    cell_capacity = full_cell[DIS_CHARGE].max()
    ano_cap = cell_capacity / (NE_hi - NE_lo)
    cat_cap = cell_capacity / (PE_hi - PE_lo)
    # Offset is (lower bound of PE * PE cap) minus (lower bound of NE * NE cap)
    offset = (z_out[0] * cat_cap) - (z_out[1] * ano_cap)
    stoic_params = [cell_capacity, cat_cap, ano_cap, offset]

    return z_out, z_cov, stoic_params


def simulate_OCV(
    anode_data, cathode_data, SOC_points, z_pe_lo, z_ne_lo, z_pe_hi, z_ne_hi
):
    """Simulate full-cell pOCV using half-cell data."""
    # Unpack the input data, make copies, and make 'z' the index column.
    pe_data = cathode_data.copy()
    pe_data["z"] = pe_data["z"].round(5)
    pe_data.set_index("z", inplace=True)
    pe_data = pe_data.loc[~pe_data.index.duplicated(), :]
    ne_data = anode_data.copy()
    ne_data["z"] = ne_data["z"].round(5)
    ne_data.set_index("z", inplace=True)
    ne_data = ne_data.loc[~ne_data.index.duplicated(), :]

    # make linearly spaced values of z for neg and pos electrode
    z_ne = np.linspace(z_ne_lo, z_ne_hi, 100001)
    z_pe = np.linspace(z_pe_lo, z_pe_hi, 100001)

    # make new dataframes for the interpolation
    ne_data_int = pd.DataFrame(data=z_ne, columns=["z"])
    ne_data_int["z"] = ne_data_int["z"].round(5)
    ne_data_int.set_index("z", inplace=True)
    ne_data_int[OCV] = ne_data[OCV]
    ne_data_int.interpolate(inplace=True)
    ne_data_int.reset_index(inplace=True)

    # same for pos electrode
    pe_data_int = pd.DataFrame(data=z_pe, columns=["z"])
    pe_data_int["z"] = pe_data_int["z"].round(5)
    pe_data_int.set_index("z", inplace=True)
    pe_data_int[OCV] = pe_data[OCV]
    pe_data_int.interpolate(inplace=True)
    pe_data_int.reset_index(inplace=True)

    # calculate full cell ocv from 1/2 cells
    cell_ocv = pd.DataFrame(data=None)
    cell_ocv[OCV] = pe_data_int[OCV] - ne_data_int[OCV]
    cell_ocv[SOC] = np.linspace(0, 1, len(cell_ocv[OCV]))
    cell_ocv.set_index(SOC, inplace=True)

    cell_out = pd.DataFrame(data=None, index=SOC_points.round(5))
    cell_out[OCV] = cell_ocv[OCV]
    cell_out.interpolate(inplace=True)
    cell_out.reset_index(inplace=True)

    return cell_out, ne_data_int, pe_data_int
