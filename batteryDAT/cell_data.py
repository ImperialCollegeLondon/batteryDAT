# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""cell_data module for defining the BatteryCell class and its methods."""

import analysis_functions as af
import dma_functions as dma
import pandas as pd
import parsers
from constants import CURRENT, DIS_CHARGE, OHM_RESISTANCE, SOC, TIME, VOLTAGE


class BatteryCell:
    """Custom class for parsing, formatting, and analysing battery data.

    The main class containing all relevant methods for parsing, formatting,
    and analysing cell-level battery data. Certain attributes can be
    assigned during initialisation, as detailed below. If required, these
    can be edited by changing the allowed_attributes variable within
    __init__, or by manually adding other attributes upon initialising. The
    allowed_attributes tuple only exists to avoid accidental assignment.

    Attributes
    ----------
    raw_data : dict
        A dictionary for holding raw data; populated using the load_data
        and add_data methods.
    processed_data : dict
        A dictionary for holding processed data; output from the data
        analysis functions will be stored in this dictionary.
    battery_cycler : str, optional
        Specify the brand of battery cycler or potentiostat used for
        collecting the associated raw data; specifying here will simplify
        parsing of data using the add_data method. If more than one battery
        cycler was used for different datasets, do not specify one here.
        The default is None.
    capacity : float, optional
        Nominal cell capacity in Ah. Used for calculations using the input
        data (for SoC etc.)
        The default value is None.
    cell_format : str, optional
        Specify the cell format (e.g. '21700', 'pouch', 'coin', etc.)
        The attribute doesn't exist unless specified.
    number_layers : int, optional
        Number of electrode layers (pairs) in the cell.
        The attribute doesn't exist unless specified.
    negative_electrode : str, optional
        Specify the cell active material used in the negative electrode,
        e.g. 'graphite', 'silicon', or 'LTO'.
        The attribute doesn't exist unless specified.
    positive_electrode : str, optional
        Specify the cell active material used in the positive electrode,
        e.g. 'NMC811', 'NCA', or 'LFP'.
        The attribute doesn't exist unless specified.
    electrolyte_solvent : str, optional
        Specify the composition of the electrolyte solvent.
        The attribute doesn't exist unless specified.
    electrolyte_salt : str, optional
        Specify the composition of the electrolyte salt.
        The attribute doesn't exist unless specified.

    """

    def __init__(self, **kwargs):
        """Init BatteryCell object."""
        allowed_attributes = (
            "capacity",
            "number_layers",
            "negative_electrode",
            "positive_electrode",
            "battery_cycler",
            "cell_format",
            "electrolyte_solvent",
            "electrolyte_salt",
        )
        default_values = {"capacity": None, "battery_cycler": None}
        self.__dict__.update(default_values)
        self.__dict__.update(
            (key, value) for key, value in kwargs.items() if key in allowed_attributes
        )
        rejected_keys = set(kwargs.keys()) - set(allowed_attributes)
        if rejected_keys:
            print(f"Argument(s) not included in attributes:\n{rejected_keys}")
        self.raw_data = {}
        self.processed_data = {}

    def __str__(self):
        """Informal string representation."""
        return_string = "Battery Type:\n"
        for key, value in self.__dict__.items():
            return_string += f"{key}: {value}\n"
        return return_string

    def __repr__(self):
        """Official string representation."""
        return_string = "BatteryCell("
        for key, value in self.__dict__.items():
            if not value:
                continue
            return_string += f"{key}={value}, "
        if len(return_string) > len("BatteryCell("):
            return_string = return_string[:-2]
        return_string += ")"
        return return_string

    @staticmethod
    def _parser_settings(batt_cycler, **kwargs):
        """Determine the parser and settings to use for data loading."""
        if not batt_cycler:
            batt_cycler = str(
                input("Type of battery cycler (biologic/maccor/basytec)?")
            )
        if batt_cycler.lower() in ["biologic", "bio-logic"]:
            parser = parsers.biologic
        elif batt_cycler.lower() == "maccor":
            parser = parsers.maccor
        elif batt_cycler.lower() == "basytec":
            parser = parsers.basytec
        else:
            print(
                """Unsupported battery cycler;
                   try "biologic", "maccor", or "basytec"."""
            )
            return
        skip_rows = kwargs.get("skip_rows")
        use_columns = kwargs.get("use_columns")
        separator = kwargs.get("separator")
        encoding = kwargs.get("encoding")
        loader_dict = {}
        if skip_rows and parser != parsers.biologic:
            loader_dict["skip_rows"] = skip_rows
        if use_columns:
            loader_dict["use_columns"] = use_columns
        if separator and parser == parsers.basytec:
            loader_dict["separator"] = separator
        if encoding and parser == parsers.basytec:
            loader_dict["encoding"] = encoding

        return parser, loader_dict

    def format_data(self, **kwargs):
        """Format existing data within the raw_data dictionary.

        A few optional functions for creating a new 'SOC (%)' column in
        the pandas dataframe containing the data, indexing the time column to
        begin at zero, and resetting the dataframe index to begin at zero.

        Parameters
        ----------
        data_name : str
            The key corresponding to the data in raw_data which is to be
            formatted.
            The default value is None, which will result in a user input box.
        create_SOC : bool, optional
            Optionally create a new column called 'SOC (%)' in the resulting
            pandas dataframe. See parsers.create_SOC for more info.
            The default is True.
        index_time : bool, optional
            Optionally reset the time column in the resulting pandas dataframe
            so that it starts at zero. See parsers.index_time for more info.
            The default is True.
        reset_index : bool, optional
            Optionally reset the index of the resulting pandas dataframe
            so that it starts at zero.
            The default is True.

        Returns
        -------
        None. The selected pandas dataframe object within raw_data is updated
        in-place.

        """
        data_name = kwargs.get("data_name")
        if not data_name:
            data_name = str(input("Key in data raw_data to be formatted:"))
        if data_name not in self.raw_data.keys():
            print(f'Cannot find "{data_name}" in raw_data')
            return
        calc_SOC = kwargs.get("create_soc", True)
        if calc_SOC:
            if isinstance(self.raw_data[data_name], list):
                for dataset in self.raw_data[data_name]:
                    dataset[SOC] = parsers.create_SOC(dataset, capacity=self.capacity)
            else:
                self.raw_data[data_name][SOC] = parsers.create_SOC(
                    self.raw_data[data_name], capacity=self.capacity
                )
        reset_time = kwargs.get("index_time", True)
        if reset_time:
            if isinstance(self.raw_data[data_name], list):
                for dataset in self.raw_data[data_name]:
                    dataset[TIME] = parsers.index_time(dataset)
            else:
                self.raw_data[data_name][TIME] = parsers.index_time(
                    self.raw_data[data_name]
                )
        reset_index = kwargs.get("reset_index", True)
        if reset_index:
            if isinstance(self.raw_data[data_name], list):
                for dataset in self.raw_data[data_name]:
                    dataset.reset_index(inplace=True, drop=True)
            else:
                self.raw_data[data_name].reset_index(inplace=True, drop=True)

    @classmethod
    def load_data(cls, filename, **kwargs):
        """Load data into a new BatteryCell object.

        This class method creates a new BatteryCell object and populates the
        raw_data dictionary with the selected datafile.
        Various optional arguments can be passed as kwargs for setting object
        attributes via the __init__ method and for selecting different
        values for the parsing statements and formatting for laoding the data.

        Parameter descriptions below are grouped by purpose (file loading and
        naming, assigning BatteryCell object attributes, parser settings, and
        data formatting).

        Parameters
        ----------
        filename : str or list
            filename(s) and directory of the file(s) to be parsed. Files must
            be exported from either biologic (.mpt), maccor (.csv) or basytec
            (.csv) cyclers.
        data_name : str, optional
            The name to give the new dataset being added to the raw_data
            dictionary (i.e. the resulting key for that dict entry).
            The default behaviour is to create a new entry called 'data_N',
            where N corresponds to the length of the raw_data dictionary.
        battery_cycler : str, optional
            Brand of the battery cycler (biologic/basytec/maccor).
            The default is None, which resorts to using self.battery_cycler.

        capacity : float, optional
            The nominal capacity of the cell in A.h.
            The default value is None.
        number_layers : str, optional
            Defines whether the cell is single or multi-layered. For
            documentation purposes in object attributes only. Only used in
            the __init__ method. Attribute does not exist without definition.
        negative_material : str, optional
            Defines the negative electrode active material. For documentation
            purposes in object attributes only. Only used in the __init__
            method. Attribute does not exist without definition.
        positive_material : str, optional
            Defines the positive electrode active material. For documentation
            purposes in object attributes only. Only used in the __init__
            method. Attribute does not exist without definition.
        cell_format : str, optional
            Defines the cell format (e.g. 'pouch', '21700'). For documentation
            purposes in object attributes only. Only used in the __init__
            method. Attribute does not exist without definition.
        electrolyte_solvent : str, optional
            Defines the electrolyte solvent. For documentation
            purposes in object attributes only. Only used in the __init__
            method. Attribute does not exist without definition.
        electrolyte_salt : str, optional
            Defines the electrolyte salt. For documentation
            purposes in object attributes only. Only used in the __init__
            method. Attribute does not exist without definition.

        skip_rows : int, optional
            Argument used in parser settings. Equivalent to 'skiprows' in
            pandas.read_csv function.
            The default is None, which will use the default values from the
            parsers module. This varies depending on which battery_cycler is
            used: maccor default = 2, basytec default = 12; biologic parser
            automatically determines the value from the header of the file.
        use_columns : list, optional
            A list of column names to read from the raw datafile.
            The default values are cycler-dependent. If values other than the
            default are used, the new columns will not be changed by the
            renaming function, and may impact the use of other methods.
        separator : str, optional
            Argument used in parser settings. Equivalent to 'sep' in
            pandas.read_csv function.
            The default values are cycler-dependent.
        encoding : str, optional
            Argument used in parser settings. Equivalent to 'encoding' in
            pandas.read_csv function.
            The default values are cycler-dependent.

        create_SOC : bool, optional
            Optionally create a new column called 'SOC (%)' in the resulting
            pandas dataframe. See parsers.create_SOC for more info.
            The default is True.
        index_time : bool, optional
            Optionally reset the time column in the resulting pandas dataframe
            so that it starts at zero. See parsers.index_time for more info.
            The default is True.
        reset_index : bool, optional
            Optionally reset the index of the resulting pandas dataframe
            so that it starts at zero.
            The default is True.

        Returns
        -------
        A BatteryCell object with the assigned attribute values and an entry in
        the raw_data dictionary corresponding to the parsed data.

        """
        battery_data = cls(**kwargs)
        data_name = kwargs.get("data_name", "data_1")
        batt_cycler = battery_data.battery_cycler
        parser, loader_dict = battery_data._parser_settings(batt_cycler, **kwargs)
        if not parser:
            return
        if isinstance(filename, str):
            filename = [filename]
        data_list = []
        for file in filename:
            data_list.append(parser(file, **loader_dict))
        battery_data.raw_data[data_name] = data_list
        kwargs.update({"data_name": data_name})
        battery_data.format_data(**kwargs)
        return battery_data

    def add_data(self, filename, **kwargs):
        """Add additional data to the raw_data dictionary.

        add_data acts in a similar way as the load_data method, but without
        creating a new BatteryCell object and without the option of assigning
        attribute values for the object.
        Optional arguments can be passed as kwargs for selecting different
        values for the parsing statements and data formatting.

        Parameters
        ----------
        filename : str
            filename(s) and directory of the file(s) to be parsed. Files must
            be exported from either biologic (.mpt), maccor (.csv) or basytec
            (.csv) cyclers.
        battery_cycler : str, optional
            Brand of the battery cycler (biologic/basytec/maccor).
            The default is None, which resorts to using self.battery_cycler.
        data_name : str, optional
            The name to give the new dataset being added to the raw_data
            dictionary (i.e. the resulting key for that dict entry).
            The default behaviour is to create a new entry called 'data_N',
            where N corresponds to the length of the raw_data dictionary.
        skip_rows : int, optional
            Argument used in parser settings. Equivalent to 'skiprows' in
            pandas.read_csv function.
            The default is None, which will use the default values from the
            parsers module. This varies depending on which battery_cycler is
            used: maccor default = 2, basytec default = 12; biologic parser
            automatically determines the value from the header of the file.
        use_columns : list, optional
            A list of column names to read from the raw datafile.
            The default values are cycler-dependent. If values other than the
            default are used, the new columns will not be changed by the
            renaming function, and may impact the use of other methods.
        separator : str, optional
            Argument used in parser settings. Equivalent to 'sep' in
            pandas.read_csv function.
            The default values are cycler-dependent.
        encoding : str, optional
            Argument used in parser settings. Equivalent to 'encoding' in
            pandas.read_csv function.
            The default values are cycler-dependent.
        create_SOC : bool, optional
            Optionally create a new column called 'SOC (%)' in the resulting
            pandas dataframe. See parsers.create_SOC for more info.
            The default is True.
        index_time : bool, optional
            Optionally reset the time column in the resulting pandas dataframe
            so that it starts at zero. See parsers.index_time for more info.
            The default is True.
        reset_index : bool, optional
            Optionally reset the index of the resulting pandas dataframe
            so that it starts at zero.
            The default is True.

        Returns
        -------
        None. The parsed data is saved as a pandas dataframe object in the
        raw_data dictionary (with key of data_name).

        """
        batt_cycler = kwargs.get("battery_cycler", self.battery_cycler)
        data_name = kwargs.get("data_name", f"data_{len(self.raw_data)}")
        kwargs.update({"data_name": data_name})
        parser, loader_dict = self._parser_settings(batt_cycler, **kwargs)
        if not parser:
            return
        if isinstance(filename, str):
            filename = [filename]
        data_list = []
        for file in filename:
            data_list.append(parser(file, **loader_dict))
        self.raw_data[data_name] = data_list
        self.format_data(**kwargs)

    def dc_resistance(self, data_name="pulse", battery_cycler=None, timestep=None):
        """Calculate resistance using R = (V2-V1)/(I2-I1).

        Can be used on any type of discharge test (CC or GITT), with the number
        of output values (rows in output df) equal to the number of 'pulses'
        (where CC is a single pulse). Alongside the 'R0' resistance, it also
        outputs the OCV and SoC values immediately prior to the pulse
        commencing. If there are no OCV rest periods between pulses (i.e. for
        a pulse-under-load test), the 'OCV' values will not actually be OCV.

        Parameters
        ----------
        data_name : str, optional
            Key in the raw_data dictionary corresponding to data for analysis.
            The default is 'pulse'.
        battery_cycler : str, optional
            Type of battery cycler used for data collection (biologic etc.)
            The default is None, resulting in self.battery_cycler being used.
        timestep : float, optional
            Float value corresponding to the time interval used for resistance
            calculation (i.e. time after pulse commencing).
            The default is None, which results in the first datapoint being
            used (~'instantaneous', depending on sampling rate of the data).

        Returns
        -------
        None. The results are written as a pandas DataFrame object to the
        self.processed_data dictionary, with the key 'R0 (Ohms)'. This will
        overwrite any previous entry.

        """
        input_data = self.raw_data[data_name]
        if not battery_cycler:
            battery_cycler = self.battery_cycler
        if not battery_cycler:
            battery_cycler = str(
                input(
                    """Which battery cycler was used?
                                       (biologic and maccor are supported)"""
                )
            )
        if battery_cycler.lower() in ["biologic", "bio-logic"]:
            if timestep:
                self.processed_data[OHM_RESISTANCE] = af.resist_time(
                    input_data, timestep
                )

            else:
                self.processed_data[OHM_RESISTANCE] = af.r0_calc_dis(input_data)
        elif battery_cycler.lower() in ["maccor"]:
            if timestep:
                print(
                    """Time-based resistance calculation only available for
                      biologic data currently."""
                )
                return None
            number_pulses = int(input("Number of current pulses in data:"))
            self.processed_data[OHM_RESISTANCE] = af.r0_calc_dis_maccor(
                input_data, number_pulses
            )
        elif battery_cycler.lower() in ["basytec"]:
            if timestep:
                print(
                    """Time-based resistance calculation only available for
                      biologic data currently."""
                )
                return None
            self.processed_data[OHM_RESISTANCE] = af.r0_calc_dis_bastyec(input_data)
        else:
            print(f"Unsupported battery cycler ({battery_cycler}).")

    def dynamic_resistance(self, data_name="CC", c_type="d"):
        """Calculate DRA profile from input dataset & reference OCV & R0 data.

        A function which takes an input dataset to be analysed (any type of
        charge or discharge, signalled by the 'c_type' argument with values of
        'c' or 'd') alongside a reference GITT dataset which contains OCV and
        R0 vs SOC data. The output is a pandas DF with 10 columns, including
        overvoltage, total resistance, dynamic resistance, etc.

        Parameters
        ----------
        data_name: str, optional
            Label of the dataset to be analysed. Must be a key from the
            raw_data dictionary.
            The default is 'CC'.
        c_type: str, optional
            Specify whether data is charge ('c') or discharge ('d').

        Returns
        -------
        None. Results are saved in self.processed_data['DRA'] as a pandas DF.


        """
        # Create copies of dataframes so originals are not affected.
        df_input = self.raw_data[data_name]
        if OHM_RESISTANCE in self.processed_data.keys():
            GITT_data = self.processed_data[OHM_RESISTANCE]
        else:
            print(
                """No ohmic resistance data found.\n
                  Please run the dc_resistance function on a GITT dataset
                  before trying again. See help for more details."""
            )
            return
        # Selecting only discharge data or only charge
        # ***PROBLEMS IF USING GITT***
        if c_type == "c":
            df_input_discharge = df_input.loc[df_input[CURRENT] > 0, :].copy()
        else:
            df_input_discharge = df_input.loc[df_input[CURRENT] < 0, :].copy()

        if data_name.lower() in ["cc", "constant", "constant current"]:
            self.processed_data["DRA"] = af.overvoltage(df_input_discharge, GITT_data)
        elif data_name.lower() in ["pulse", "pulsed", "gitt", "hppc"]:
            self.processed_data["DRA"] = af.overvoltage_pulse(
                df_input_discharge, GITT_data
            )
        else:
            print('Unsupported data_name; try "cc" or "pulse".')

    def dQdV(self, data_name=None, dV_range=0.005, V_total=1.700, I_type="d"):
        """Calculate the Incremental Capacity Analysis (ICA or dQ/dV) profile.

        A function for calculating dQ/dV data from a pandas DF containing
        "SOC", "CURRENT", and "VOLTAGE" columns. The function can calculate
        dQ/dV for discharge ('d'), charge ('c') or both ('b'), outputting
        either a single DF or a list of 2 DFs. Each DF contains "SOC",
        "VOLTAGE", and "dQ/dV" columns. The function uses a finite-difference
        method, with default dV values of 5 mV with a total range of 1700 mV
        (i.e. 4.2 to 2.5 V).

        The function is more complicated due to the possibility of having
        discharge, charge, or both. The opertions themselves are very simple
        though.

        Parameters
        ----------
        data_name: str, optional
            Label of the dataset to be analysed. Must be a key from the
            raw_data dictionary.
            The default is 'data_1'.
        dV_range : float, optional
            Finite step size for dV in the dQ/dV calculation. Lower numbers
            give greater resolution, but also greater noise.
            The default is 0.005 (i.e. 5 mV).
        V_total : float, optional
            Total voltage range for the analysis, i.e. maximum voltage minus
            minimum voltage. E.g. a discharge from 4.2 V to 2.5 V would be a
            range of 1.7 V.
            The default is 1.700.
        I_type: str, optional
            Label corresponding to charge/discharge type. Can be 'd', 'c', or
            'b', corresonding to discharge, charge, or both.

        Returns
        -------
        None. Results are saved in self.processed_data['ICA'] as a pandas DF.

        """
        if not data_name:
            data_name = "data_1"
        if data_name not in self.raw_data.keys():
            print("Cannot find data_name in raw_data dictionary.")
            return None

        self.processed_data["ICA"] = af.dQdV(
            self.raw_data[data_name], dV_range=dV_range, V_total=V_total, I_type=I_type
        )

    def dRdQ(self, dQ_range=1, Q_total=100):
        """Calculate differential of dynamic resistance w.r.t. charge passed.

        A function for calculating the differential resistance (dR/dQ).
        This function requires an entry called 'DRA' exists in the
        processed_data dictionary (i.e. output from the dynamic_resistance
        function. The function uses a finite-difference method with default dQ
        (or dSOC) value of 1%.
        The output is a DF with 'SOC' and 'dR/dSOC (Ohms/%)' columns.

        Parameters
        ----------
        dQ_range : float, optional
            Finite step size for dQ in the dR/dQ calculation. Lower numbers
            give greater resolution, but also greater noise.
            The default is 1.
        Q_total : float, optional
            Number of bins used to split the total charge passed into. Higher
            numbers give greater resolution, but also greater noise.
            The default is 100.

        Returns
        -------
        None. Results are saved in self.processed_data['dDRA'] as a pandas DF.

        """
        if "DRA" not in self.processed_data.keys():
            print("No DRA data found in the processed_data dictionary.")
            return None

        self.processed_data["dDRA"] = af.dRdQ_calc(
            self.processed_data["DRA"], dQ_range, Q_total=Q_total
        )

    def OCV_fit(
        self,
        data_name=None,
        age=0,
        composite=False,
        positive_electrode=None,
        negative_electrode=None,
        negative_electrode_2=None,
        guess_values=None,
        diff_step=None,
    ):
        """Determine electrode SoC limits from pOCV data.

        Function for calculating the operational SoC limits (and therefore
        capacities and offset) of the individual electrodes when discharging
        the cell. Uses pseudo-OCV (pOCV) data from the full cell (at any
        state-of-health) alongside reference beginning-of-life pOCV data for
        each electrode material.

        Returns the fitting parameters from the optimisation function as well
        as the calculated electrode capacities and offsets.

        For full details of the fitting function parameters, see
        dma_functions.stoich_OCV_fit_multi_comp().

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset in raw_data to be analysed.
            The default is None, which results in a user input being required.
        age : int, optional
            An integer value representing the 'age' of the data in case there
            is more than one dataset present in raw_data with the same
            data_name name. Expected use-case is where a user has multiple
            pOCV datasets at different states-of-health, all saved within a
            list with the same data_name name. The default is 0.
        composite : Bool, optional
            Define whether the negative electrode is a composite or not.
            The default is False.
        positive_electrode : str, optional
            Filename (and path if not in the working directory) of the PE data.
            This should be a csv file containing the 1/2 cell data for the
            positive electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode : str, optional
            Filename (and path if not in the working directory) of the NE data.
            This should be a csv file containing the 1/2 cell data for the
            negative electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode_2 : str, optional
            Filename (and path if not in the working directory) of the second
            NE dataset, only required if composite = True (i.e. there are two
            negative electrode active materials). This should be a csv file
            containing the 1/2 cell data for the second negative electrode
            active material; this data should contain voltage and capacity.
            The default is None, which results in a user input being required
            (if composite = True).
        guess_values : list, optional
            A list of floats between 0 and 1, representing initial guess values
            for the fitting parameters. If composite = False, there are 4
            values which represent the lower and upper SoC fractions for each
            electrode [PE_lo, NE_lo, PE_hi, NE_hi]. If composite = True, there
            is an additional value corresponding to the capacity fraction of
            component 1 of the negative electrode.
            The default is [0.1, 0.002, 0.95, 0.85, 0.84].
        diff_step : float, optional
            Controls the step size between iterations of the optimisation
            function. Smaller step sizes result in finer tuning of the params,
            but increases the risks of getting stuck in a local minima and can
            result in slower calculation. The default is 0.01.

        Returns
        -------
        None. Results are saved in self.processed_data['OCV-fit'][age] in the
        form of two pandas dataframe objects (parameters and capacities).

        """
        # Check full-cell data selected for OCV-fitting exists etc.
        if not data_name:
            data_name = str(input("Select dataset to use for OCV-fitting"))
        if data_name in self.raw_data.keys():
            if isinstance(self.raw_data[data_name], list):
                input_data = self.raw_data[data_name][age].copy()
            elif isinstance(self.raw_data[data_name], pd.core.frame.DataFrame):
                input_data = self.raw_data[data_name].copy()
            else:
                print("Error: selected data is not a pandas dataframe.")
        else:
            print("Could not find this data. Please try again.")
        # Format full-cell data for use in OCV-fitting function.
        input_data = input_data[input_data[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
        input_data.reset_index(inplace=True, drop=True)
        input_data[SOC] = 1 - (input_data[DIS_CHARGE] / input_data[DIS_CHARGE].max())
        if input_data[SOC].iloc[-1] > input_data[SOC].iloc[-2]:
            input_data = input_data.iloc[:-1, :]

        # Format PE data for OCV-fitting function.
        PE_data = dma.check_electrode_data(positive_electrode, "positive electrode")
        if not isinstance(PE_data, pd.core.frame.DataFrame):
            print("Error: try again with different positive electrode data.")
            return

        # If NE is not a composite, format NE data then do 'normal' OCV-fit.
        if not composite:
            NE_data = dma.check_electrode_data(negative_electrode, "negative electrode")
            if not isinstance(NE_data, pd.core.frame.DataFrame):
                print("Error: try again with different NE data.")
                return

            if "OCV-fit" not in self.processed_data.keys():
                self.processed_data["OCV-fit"] = {}

            fit_params, _, fit_capacities = dma.stoich_OCV_fit(
                NE_data, PE_data, input_data, guess_values, diff_step
            )
        # If NE is a composite, format both NE data and do composite OCV-fit.
        else:
            NE_data_1 = dma.check_electrode_data(negative_electrode, "NE component 1")
            if not isinstance(NE_data_1, pd.core.frame.DataFrame):
                print("Error: try again with different NE comp. 1 data.")
                return
            NE_data_2 = dma.check_electrode_data(negative_electrode_2, "NE component 2")
            if not isinstance(NE_data_2, pd.core.frame.DataFrame):
                print("Error: try again with different NE comp. 2 data.")
                return

            if "OCV-fit" not in self.processed_data.keys():
                self.processed_data["OCV-fit"] = {}

            fit_params, _, fit_capacities = dma.stoich_OCV_fit_multi_comp(
                NE_data_1, NE_data_2, PE_data, input_data, guess_values, diff_step
            )
        # Add OCV-fit result to the processed_data dict, with 'age' as the key.
        self.processed_data["OCV-fit"][age] = fit_params, fit_capacities

    def DMA(
        self,
        data_name=None,
        composite=False,
        positive_electrode=None,
        negative_electrode=None,
        negative_electrode_2=None,
        carry_guess=True,
        guess_values=None,
        diff_step=None,
    ):
        """Perform degradation mode analysis on pOCV discharge data.

        Method for quantifying degradation modes of a cell as a function of
        age. Operates in the same manner as the self.OCV_fit method, except it
        takes full-cell pOCV data at more than one state-of-health, allowing
        the change in electrode capacities and offset to be calculated. As
        such, one of the input datasets should be at beginning-of-life to allow
        for comparison against.

        Returns the quantified degradation modes and calculated electrode
        capacities and offsets, all as a function of age.

        For further details of the fitting function parameters, see the
        self.OCV_fit method and the dma_functions.DM_calc_multi_comp function.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset in raw_data to be analysed.
            The default is None, which results in a user input being required.
        composite : Bool, optional
            Define whether the negative electrode is a composite or not.
            The default is False.
        positive_electrode : str or pd.dataFrame, optional
            Filename (and path if not in the working directory) of the PE data.
            This should be a csv file containing the 1/2 cell data for the
            positive electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode : str or pd.dataFrame, optional
            Filename (and path if not in the working directory) of the NE data.
            This should be a csv file containing the 1/2 cell data for the
            negative electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode_2 : str or pd.dataFrame, optional
            Filename (and path if not in the working directory) of the second
            NE dataset, only required if composite = True (i.e. there are two
            negative electrode active materials). This should be a csv file
            containing the 1/2 cell data for the second negative electrode
            active material; this data should contain voltage and capacity.
            The default is None, which results in a user input being required
            (if composite = True).
        carry_guess : bool, optional
            Specify whether the fitted parameters from one optimisation should
            be carried forward as the initial guess for the next SoH.
            Default value is True.
        guess_values : list, optional
            A list of floats between 0 and 1, representing initial guess values
            for the fitting parameters. If composite = False, there are 4
            values which represent the lower and upper SoC fractions for each
            electrode [PE_lo, NE_lo, PE_hi, NE_hi]. If composite = True, there
            is an additional value corresponding to the capacity fraction of
            component 1 of the negative electrode.
            The default is [0.1, 0.002, 0.95, 0.85, 0.84].
        diff_step : float, optional
            Controls the step size between iterations of the optimisation
            function. Smaller step sizes result in finer tuning of the params,
            but increases the risks of getting stuck in a local minima and can
            result in slower calculation. The default is 0.01.

        Returns
        -------
        None. Results are saved in self.processed_data['DMA'] in the
        form of two pandas dataframe objects (DMs and capacities).

        """
        # Check full-cell data selected for OCV-fitting exists etc.
        if not data_name:
            data_name = str(input("Select which datasets to use for DMA"))
        if data_name in self.raw_data.keys():
            if not isinstance(self.raw_data[data_name], list):
                print("Error: selected data isn't a list of pandas dataframes.")
            else:
                if not isinstance(self.raw_data[data_name][0], pd.core.frame.DataFrame):
                    print("""Error: list doesn't contain pandas dataframes.""")
        else:
            print("Could not find this data. Please try again.")

        # Format PE data for OCV-fitting function.
        PE_data = dma.check_electrode_data(positive_electrode, "positive electrode")
        if not isinstance(PE_data, pd.core.frame.DataFrame):
            print("Error: try again with different positive electrode data.")
            return

        # If NE is not a composite, format NE data then do 'normal' OCV-fit.
        if not composite:
            print("Only composite DMA currently supported. See old version.")
            return
            NE_data = dma.check_electrode_data(negative_electrode, "negative electrode")
            if not isinstance(NE_data, pd.core.frame.DataFrame):
                print("Error: try again with different NE data.")
                return
            # UPDATE WITH NON-COMPOSITE VERSION
            DMs, caps, _, _ = dma.DM_calc_multi_comp_long(
                NE_data,
                PE_data,
                self.raw_data[data_name][0],
                self.raw_data[data_name][1:],
                carry_guess=carry_guess,
            )
        # If NE is a composite, format both NE data and do composite OCV-fit.
        else:
            NE_data_1 = dma.check_electrode_data(negative_electrode, "NE component 1")
            if not isinstance(NE_data_1, pd.core.frame.DataFrame):
                print("Error: try again with different NE comp. 1 data.")
                return
            NE_data_2 = dma.check_electrode_data(negative_electrode_2, "NE component 2")
            if not isinstance(NE_data_2, pd.core.frame.DataFrame):
                print("Error: try again with different NE comp. 2 data.")
                return

            DMs, caps, _, _ = dma.DM_calc_multi_comp_long(
                NE_data_1,
                NE_data_2,
                PE_data,
                self.raw_data[data_name][0],
                self.raw_data[data_name][1:],
                carry_guess=carry_guess,
            )
        self.processed_data["DMA"] = DMs, caps

    def OCV_sim(
        self,
        data_name=None,
        age=0,
        composite=False,
        positive_electrode=None,
        negative_electrode=None,
        negative_electrode_2=None,
        fitting_parameters=None,
    ):
        """Calculate an OCV profile based on some given electrode parameters.

        Takes positive and negative electrode pOCV data alongside upper &
        lower lithiation limits for each electrode and calculates a full-cell
        pOCV profile. If the negative electrode is a composite material (i.e.,
        more than one active material), a negative electrode pOCV profile will
        also be calculated based on the capacity fraction of the two materials.
        In this case, pOCV data for each component of the negative electrode
        must be supplied.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset in raw_data to be used as the SoC basis for the
            simulation. i.e., the pOCV data which the simulated pOCV can be
            compared against. The default is None, which results in a user
            input being required.
        age : int, optional
            Integer value corresponding to the list index where the fitting
            parameters are located. Only required if fitting parameters are
            taken from results of a previous OCV_fit or DMA method call, saved
            in self.processed_data['OCV-fit'][age][0]. The default is 0.
        composite : Bool, optional
            Select whether the negative electrode is a composite material or
            not. The default is False.
        positive_electrode : str, optional
            Filename (and path if not in the working directory) of the PE data.
            This should be a csv file containing the 1/2 cell data for the
            positive electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode : str, optional
            Filename (and path if not in the working directory) of the NE data.
            This should be a csv file containing the 1/2 cell data for the
            negative electrode; this data should contain voltage and capacity.
            The default is None, which results in a user input being required.
        negative_electrode_2 : str, optional
            Filename (and path if not in the working directory) of the second
            NE dataset, only required if composite = True (i.e. there are two
            negative electrode active materials). This should be a csv file
            containing the 1/2 cell data for the second negative electrode
            active material; this data should contain voltage and capacity.
            The default is None, which results in a user input being required
            (if composite = True).
        fitting_parameters : list, optional
            A list of floats between 0 and 1, representing the fitting
            parameters. If composite = False, there are 4 values which
            represent the lower and upper SoC fractions for each electrode
            [PE_lo, NE_lo, PE_hi, NE_hi]. If composite = True, there is an
            additional value corresponding to the capacity fraction of
            component 1 of the negative electrode. The default behaviour takes
            values from self.processed_data['OCV-fit'][age][0].

        Returns
        -------
        None. Output datasets stored in self.processed_data['simulation'][age],
        consisting of a tuple which contains the simulated full-cell, negative
        electrode, and positive electrode data.

        """
        # Check if fitting parameters exist. Exit function if not.
        if not fitting_parameters:
            if "OCV-fit" not in self.processed_data.keys():
                print(
                    """No OCV-fitting parameters found in processed_data.
                      Please run the OCV_fit method to obtain fitting
                      parameters for simulation."""
                )
                return
            else:
                fitting_parameters = self.processed_data["OCV-fit"][age][0]

        # Find data.
        if not data_name:
            data_name = str(input("Define which dataset to use as SoC basis for sim"))
        if data_name in self.raw_data.keys():
            if isinstance(self.raw_data[data_name], list):
                input_data = self.raw_data[data_name][age].copy()
            elif isinstance(self.raw_data[data_name], pd.core.frame.DataFrame):
                input_data = self.raw_data[data_name].copy()
            else:
                print("Error: selected data is not a pandas dataframe.")
        else:
            print("Could not find this data. Please try again.")
        # Format full-cell data for use in OCV-fitting function.
        input_data = input_data[input_data[CURRENT] < 0].loc[:, [DIS_CHARGE, VOLTAGE]]
        input_data.reset_index(inplace=True, drop=True)
        input_data[SOC] = 1 - (input_data[DIS_CHARGE] / input_data[DIS_CHARGE].max())
        if input_data[SOC].iloc[-1] > input_data[SOC].iloc[-2]:
            input_data = input_data.iloc[:-1, :]

        # Format PE data for OCV-fitting function.
        PE_data = dma.check_electrode_data(positive_electrode, "positive electrode")
        if not isinstance(PE_data, pd.core.frame.DataFrame):
            print("Error: try again with different positive electrode data.")
            return

        # If NE is not a composite, format NE data then do 'normal' OCV-fit.
        if not composite:
            NE_data = dma.check_electrode_data(negative_electrode, "negative electrode")
            if not isinstance(NE_data, pd.core.frame.DataFrame):
                print("Error: try again with different NE data.")
                return

            cell_sim, NE_sim, PE_sim = dma.simulate_OCV(
                NE_data, PE_data, input_data[SOC], *fitting_parameters
            )

        if "simulation" not in self.processed_data.keys():
            self.processed_data["simulation"] = {}

        self.processed_data["simulation"][age] = cell_sim, NE_sim, PE_sim
