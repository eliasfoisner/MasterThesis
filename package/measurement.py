from . import utils
import pandas as pd
import numpy as np
import scipy as sc
import plotly.graph_objects as go


class SyngistixMeasurement:
    """
    SyngistixMeasurement
    A class for processing and analyzing Syngistix and Nanomodul measurement files, providing methods for timescale calculation, signal smoothing, background correction, peak detection, peak characterization, area integration, calibration, area ratio calculation, and visualization.
    Attributes:
        file_name (str): Name of the input file.
        data (pd.DataFrame): DataFrame containing measurement data.
        measured_isotopes (pd.Index): List of measured isotopes.
        peaks (dict): Dictionary storing peak properties for each isotope.
    Methods:
        __init__(file_path: str):
            Initialize the measurement object, load data, and determine file type (Syngistix or Nanomodul).
        timescale(isotope: str, cycle_time: float = False):
            Add a timescale column for a specific isotope. Requires cycle_time for Nanomodul files.
        savgol(isotope: str, window_length: int = 100, polyorder: int = 2, deriv: int = 0):
            Apply Savitzky-Golay smoothing to the isotope signal.
        global_background(isotope: str, start: float = 0.1, end: float = 1):
            Calculate and subtract a global background from the isotope signal.
        peak_finding(isotope: str, threshold: float = 50, distance: float = 10e-3, width: float = 10e-6, savgol: bool = False):
            Detect peaks in the isotope signal using scipy.signal.find_peaks().
        peak_width(isotope: str, criterion: int = 10):
            Calculate peak widths using scipy.signal.peak_widths().
        peak_background(isotope: str, distance: float = 0, window_size: float = 3):
            Estimate local background around each peak.
        peak_area(isotope: str, mode: str = 'trapezoid', resize: float = 1, local_background: bool = False):
            Integrate the area under each peak, with optional local background correction.
        calibrate(isotope, calibration: object):
            Calibrate peak areas to mass using a provided calibration object.
        area_ratio(isotope_one: str, gravfac_one: float, isotope_two: str, gravfac_two: float):
            Calculate area ratios between two isotopes for each peak.
        plot(isotope: str, fig: object, savgol: bool = False, integration: bool = False, peaks: bool = False, background: bool = False, width: bool = False):
            Plot the isotope signal and optional features such as peaks, backgrounds, and integration windows.
    """

    def __init__(self, file_path: str):
        """
        Initializes the measurement object by loading data from the specified file path.
        Parameters:
            file_path (str): The path to the data file to be loaded
            
        Attributes:
            file_name (str): The name of the data file.
            __timescale_count (int): Counter for timescale, initialized to 0.
            __nanomodul (bool): Indicates if the file is in nanomodul format.
            data (pd.DataFrame): The loaded measurement data.
            measured_isotopes (Index or list): The names of the measured isotopes.
            peaks (dict): A dictionary mapping each isotope to an empty DataFrame for storing peak information.
        """
        self.file_name = file_path.name
        self.__timescale_count = 0
        self.__nanomodul = utils.check_nanomodul(file_path)
        if self.__nanomodul:
            self.data = pd.read_csv(file_path, skiprows=0).iloc[:, :-1]
            self.measured_isotopes = self.data.columns
        else:
            self.data = pd.read_csv(file_path, skiprows=2)
            self.measured_isotopes = self.data.loc[:, self.data.columns != "Time in Seconds "].columns
        self.peaks = dict()
        for isotope in self.measured_isotopes:
            self.peaks[isotope] = pd.DataFrame(columns=['index', 'time', 'height', 'width', 'background', 'area'])

    def timescale(self, isotope: str, cycle_time: float = False):
        """
        Generates and assigns a time scale for the specified isotope in the measurement data.

        Parameters:
            isotope (str): The name of the isotope for which to generate the time scale. Must be present in self.measured_isotopes.
            cycle_time (float, optional): The cycle time to use for time scale calculation. Required if using Nanomodul files (self.__nanomodul is True).

        Returns:
            bool: True if the time scale was successfully generated and assigned.

        Raises:
            AssertionError: If the specified isotope is not in self.measured_isotopes.
            AssertionError: If using Nanomodul files and cycle_time is not specified.

        Notes:
            - For Nanomodul files, the method requires an explicit cycle_time and calculates timestamps accordingly.
            - For other files, it infers the cycle time from the data and generates a linearly spaced time scale.
            - The generated time scale is stored in self.data under the key f"{isotope}_time".
        """
        assert isotope in self.measured_isotopes, f"Isotope '{isotope}' was not measured! {self.file_name}"
        if self.__nanomodul:
            assert cycle_time, "Please specify cycle time when using Nanomodul files!"
            self.__cycle_time = cycle_time
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return True
        else:
            self.data["Time in Seconds "] = np.linspace(0, max(self.data["Time in Seconds "]), len(self.data))
            self.__cycle_time = self.data["Time in Seconds "][1] - self.data["Time in Seconds "][0]
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return True

    def savgol(self, isotope: str, window_length: int = 100, polyorder: int = 2, deriv: int = 0):
        """
        Applies a Savitzky-Golay filter to the specified isotope data and stores the result.

        Parameters:
            isotope (str): The key corresponding to the isotope data column in self.data to be filtered.
            window_length (int, optional): The length of the filter window (i.e., the number of coefficients). Must be a positive odd integer. Default is 100.
            polyorder (int, optional): The order of the polynomial used to fit the samples. Must be less than window_length. Default is 2.
            deriv (int, optional): The order of the derivative to compute. Default is 0 (no differentiation).

        Returns:
            bool: True if the filter was applied and the result was stored successfully.
        """

        self.data[f"{isotope}_savgol"] = sc.signal.savgol_filter(self.data[isotope], window_length=window_length, polyorder=polyorder, deriv=deriv)
        return True

    def global_background(self, isotope: str, start: float = 0.1, end: float = 1):
        """
        Calculates and subtracts a global background value for a specified isotope over a given time range.

        Parameters:
            isotope (str): The name of the isotope to process.
            start (float, optional): The start time (in seconds) for the background calculation window. Defaults to 0.1.
            end (float, optional): The end time (in seconds) for the background calculation window. Defaults to 1.

        Returns:
            bool: True if the background correction was applied successfully.

        Side Effects:
            - Sets the attribute `self.__global_bg` to the computed median background value.
            - Adds a new column to `self.data` named '{isotope}_corr' containing the background-corrected data.
        """
        start_idx = int(start // self.__cycle_time)
        end_idx = int(end // self.__cycle_time)
        self.__global_bg = np.median(self.data[isotope][start_idx:end_idx])
        self.data[f"{isotope}_corr"] = self.data[f"{isotope}"] - self.__global_bg
        return True

    def peak_finding(self, isotope: str, threshold: float = 50, distance: float = 10e-3, width: float = 10e-6, savgol: bool = False):
        """
        Detects peaks in the signal data for a specified isotope using configurable parameters.

        Parameters:
            isotope (str): The name of the isotope to analyze. Must have corresponding '{isotope}_time' column in self.data.
            threshold (float, optional): Minimum height required for a peak to be detected. Defaults to 50.
            distance (float, optional): Minimum horizontal distance (in seconds) between neighboring peaks. Defaults to 10e-3.
            width (float, optional): Required width (in seconds) of peaks. Defaults to 10e-6.
            savgol (bool, optional): If True, applies Savitzky-Golay smoothing before peak detection. Defaults to False.

        Raises:
            AssertionError: If the required '{isotope}_time' column is not present in self.data.

        Returns:
            bool: True if peak detection was performed successfully.

        Side Effects:
            Updates self.peaks[isotope] with detected peak indices, times, and heights.
        """
        assert f'{isotope}_time' in self.data.columns, f"Calculate '{isotope}_time' before peak detection!"
        if savgol:
            self.savgol()
            signal = self.data[f"{isotope}_savgol"]
        else:
            signal = self.data[f"{isotope}"]
            self.peaks[isotope]['index'] = sc.signal.find_peaks(signal, height=threshold, distance=distance/self.__cycle_time, width=width/self.__cycle_time)[0]

        peak_idx = self.peaks[isotope]['index']
        self.peaks[isotope]['time'] = self.data[f'{isotope}_time'].iloc[peak_idx].tolist()
        self.peaks[isotope]['height'] = self.data[isotope].iloc[peak_idx].tolist()
        return True
    
    def peak_width(self, isotope: str, criterion: int = 10):
        """
        Calculates the width of detected peaks for a given isotope using a specified criterion.

        Parameters:
            isotope (str): The name of the isotope for which to calculate peak widths.
            criterion (int, optional): The percentage criterion for peak width calculation (default is 10).
                The width is measured at (1 - criterion/100) of the peak height.

        Returns:
            bool: True if the peak widths are successfully calculated and stored.

        Side Effects:
            Updates the 'width', 'time_left', and 'time_right' fields in the self.peaks[isotope] dictionary
            with the calculated peak widths and their corresponding left and right time positions.

        Raises:
            AssertionError: If the calculated left and right peak width arrays have different lengths.
        """

        width = sc.signal.peak_widths(x=self.data[f'{isotope}'], peaks=self.peaks[isotope][f'index'], rel_height=(1 - criterion / 100))

        peakwidth = width[0] * self.__cycle_time
        left = [i*self.__cycle_time for i in width[2]]
        right = [i*self.__cycle_time for i in width[3]]

        assert len(left) == len(right), f"Left and right peak width arrays have different lengths! {self.file_name}"
        self.peaks[isotope]['width'] = peakwidth
        self.peaks[isotope]['time_left'] = left
        self.peaks[isotope]['time_right'] = right

        return True

    def peak_background(self, isotope: str, distance: float = 0, window_size: float = 3):
        """
        Calculates and assigns the background signal for each detected peak of a given isotope.

        For each peak, the method determines two background regions (left and right) based on the peak's position,
        width, and asymmetry. The mean values of these regions are computed and summed to estimate the background
        for each peak. The results are stored in `self.peaks[isotope]['background']`.

        Args:
            isotope (str): The isotope for which to calculate peak backgrounds.
            distance (float, optional): Factor to determine the distance of the background window from the peak center. Defaults to 0.
            window_size (float, optional): Size of the background window (in units of peak width). Defaults to 3.

        Returns:
            bool: True if background calculation was successful.
        """

        backgrounds = []
        self.__bgwindowsize = window_size
        self.__bgfactor = distance

        for i in range(len(self.peaks[isotope]['index'])):
            peak_width = self.peaks[isotope]['width'][i]
            peak_time = self.peaks[isotope]['time'][i]
            peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            bgbound_left_outer_idx = (peak_time - peak_width * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)) / self.__cycle_time
            bgbound_left_inner_idx = (peak_time - peak_width * self.__bgfactor * (1 - asymmetry)) / self.__cycle_time
            left_data = self.data[isotope][int(bgbound_left_outer_idx):int(bgbound_left_inner_idx)]

            bgbound_right_inner_idx = (peak_time + peak_width * self.__bgfactor * asymmetry) / self.__cycle_time
            bgbound_right_outer_idx = (peak_time + peak_width * (self.__bgfactor * asymmetry + self.__bgwindowsize)) / self.__cycle_time
            right_data = self.data[isotope][int(bgbound_right_inner_idx):int(bgbound_right_outer_idx)]

            backgrounds.append((left_data.mean(), right_data.mean()))

        self.peaks[isotope]['background'] = [i+j for i, j in backgrounds]

        return True

    def peak_area(self, isotope: str, mode: str = 'trapezoid', resize: float = 1, local_background: bool = False):
        """
        Calculates the area under the peaks for a given isotope using numerical integration.

        Parameters:
            isotope (str): The isotope for which to calculate peak areas.
            mode (str, optional): The integration method to use. Options are 'trapezoid' (default) or 'romberg'.
            resize (float, optional): Factor to resize the integration bounds. Default is 1.
            local_background (bool, optional): If True, subtracts local background for each peak; otherwise, subtracts global background. Default is False.

        Returns:
            bool: True if the calculation was successful. The calculated areas are stored in self.peaks[isotope]['area'].

        Notes:
            - Integration bounds are determined based on peak position, width, and asymmetry.
            - Requires that peak information (time, width, background, etc.) is already populated in self.peaks[isotope].
            - Uses self.data[isotope] for signal values and self.__cycle_time for time-to-index conversion.
        """
        self.__intfactor = resize
        areas = []
        for i in range(len(self.peaks[isotope]['index'])):
            peak_time = self.peaks[isotope]['time'][i]
            peak_width = self.peaks[isotope]['width'][i]
            peak_background = self.peaks[isotope]['background'][i]
            peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            intbound_left_idx = int((peak_time - peak_width * self.__intfactor * (1 - asymmetry)) / self.__cycle_time)
            intbound_right_idx = int((peak_time + peak_width * self.__intfactor * asymmetry) / self.__cycle_time)

            if local_background:
                #assert peak_time.empty is False and not peak_background.isna().all(), f"Calculate local background before integration or set local_background = False!"
                signal = [k - self.peaks[isotope]['background'][i] for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]
            else:
                signal = [k - self.global_background(isotope, start=0.1, end=1) for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]

            if mode == 'trapezoid':
                areas.append(sc.integrate.trapezoid(signal))
            elif mode == 'romberg':
                areas.append(sc.integrate.romberg(signal))

        self.peaks[isotope]['area'] = areas
        return True


    def area_ratio(self, isotope_one: str, gravfac_one: float, isotope_two: str, gravfac_two: float):
        """
        Calculates the area ratio between two isotopes within specified time windows.

        For each peak of `isotope_one`, this method computes the area under the curve for `isotope_two`
        within the same time window as the peak of `isotope_one`. It then calculates the ratio:
            ratio = (area_one * gravfac_one) / (area_one * gravfac_one + area_two * gravfac_two)
        and collects all such ratios (excluding those equal to 1).

        Args:
            isotope_one (str): The name of the first isotope (used to select peaks and time windows).
            gravfac_one (float): Gravimetric factor for the first isotope.
            isotope_two (str): The name of the second isotope (used to extract signal data).
            gravfac_two (float): Gravimetric factor for the second isotope.

        Returns:
            tuple: A tuple containing:
                - ratios (list of float): List of calculated ratios for each peak.
                - ratio_mean (float): Mean of the calculated ratios.
                - ratio_stdev (float): Standard deviation of the calculated ratios.
        """
        ratios = []
        for i_one in range(len(self.peaks[isotope_one])):
            area_one = self.peaks[isotope_one]["area"][i_one]

            left_timestamp = self.peaks[isotope_one]["time_left"][i_one]
            right_timestamp = self.peaks[isotope_one]["time_right"][i_one]

            signal = self.data[(left_timestamp < self.data["Time in Seconds "]) & (self.data["Time in Seconds "] < right_timestamp)][isotope_two]

            area_two = sc.integrate.trapezoid(signal)

            ratio = area_one * gravfac_one / (area_one * gravfac_one + area_two * gravfac_two)
            if ratio != 1:
                ratios.append(ratio)

        ratio_mean = np.mean(ratios)
        ratio_stdev = np.std(ratios)

        return (ratios, ratio_mean, ratio_stdev)
    
    def plot(self, isotope: str, fig: object, savgol: bool = False, integration: bool = False, peaks: bool = False, background: bool = False, width: bool = False):
        """
        Plots the signal and various analysis results for a given isotope using Plotly.

        Args:
            isotope (str): The isotope to plot (column prefix in self.data).
            fig (object): A Plotly Figure object to which traces and shapes will be added.
            savgol (bool, optional): If True, plot the Savitzky-Golay smoothed signal (requires prior computation).
            integration (bool, optional): If True, plot integration boundaries for each detected peak.
            peaks (bool, optional): If True, plot markers at detected peak positions.
            background (bool, optional): If True, plot background estimation lines for each peak.
            width (bool, optional): If True, plot rectangles indicating the width of each detected peak.

        Raises:
            AssertionError: If required data columns are missing or if prerequisite methods have not been called before plotting certain features.

        Notes:
            - The method updates the provided Plotly figure in-place by adding traces and shapes.
            - Requires that peak detection and related computations (e.g., Savitzky-Golay, background, width) have been performed prior to plotting those features.
        """

        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Intensity (cts)")
        fig.update_traces(line_width=1, selector=dict(type='scatter'), showlegend=True)

        traces = []
        shapes = []

        assert f'{isotope}_time' in self.data.columns, f"'{isotope}_time' required!"
        original_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}"], mode='lines', name=f"{isotope} ({self.file_name})")
        traces.append(original_trace)

        if savgol:
            assert f'{isotope}_savgol' in self.data.columns, f"Use savgol() before plotting Savitzky-Golay signal!"
            savgol_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}_savgol"], mode='lines', name=f"'{self.file_name}' ({isotope}_savgol)")
            traces.append(savgol_trace)

        try:
            peak_index = self.peaks[isotope]['index']
            peak_time = self.peaks[isotope]['time']
            peak_height = self.peaks[isotope]['height']
            peak_width = self.peaks[isotope]['width']
            peak_background = self.peaks[isotope]['background']
        except:
            pass

        if peaks:
            peak_trace = go.Scatter(x=peak_time, y=peak_height, mode='markers', marker_color="red", name=f"{isotope} peaks ({self.file_name})")
            traces.append(peak_trace)

        for i in range(len(peak_index)):
            if width:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting peak widths!"
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                width_shape = dict(type='rect', x0=peak_time_left, x1=peak_time_right, y0=0, y1=peak_height[i], line_width=0, fillcolor='blue', opacity=0.1)
                shapes.append(width_shape)

            if background:
                assert peak_time.empty is False and not peak_background.isna().all(), "Use peak_finding() and peak_background() before plotting backgrounds!"
                bgleft, bgright = self.__backgrounds[i]
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                bgbound_left_outer_time = peak_time[i] - peak_width[i] * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)
                bgbound_left_inner_time = peak_time[i] - peak_width[i] * self.__bgfactor * (1 - asymmetry)
                bgbound_right_inner_time = peak_time[i] + peak_width[i] * self.__bgfactor * asymmetry
                bgbound_right_outer_time = peak_time[i] + peak_width[i] * (self.__bgfactor * asymmetry + self.__bgwindowsize)
                bg_shape_left = dict(type="line", y0=bgleft, y1=bgleft, x0=bgbound_left_outer_time, x1=bgbound_left_inner_time, line=dict(color="red", width=3))
                bg_shape_right = dict(type="line", y0=bgright, y1=bgright, x0=bgbound_right_inner_time, x1=bgbound_right_outer_time, line=dict(color="red", width=3))
                shapes.append(bg_shape_left)
                shapes.append(bg_shape_right)

            if integration:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting integration boundaries!"
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                intbound_left_time = peak_time[i] - peak_width[i] * (1 - asymmetry) * self.__intfactor
                intbound_right_time = peak_time[i] + peak_width[i] * asymmetry * self.__intfactor
                width_shape = dict(type='rect', x0=intbound_left_time, x1=intbound_right_time, y0=0, y1=peak_height[i], line_width=0,
                                   fillcolor='green', opacity=0.1)
                shapes.append(width_shape)

        for trace in traces:
            fig.add_trace(trace)
        for shape in shapes:
            fig.add_shape(shape)