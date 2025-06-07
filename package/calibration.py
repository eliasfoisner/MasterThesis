from . import utils
import pandas as pd
import numpy as np
import scipy as sc
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import t
from scipy.optimize import curve_fit



class SyngistixCalibration:
    """
    SyngistixCalibration provides methods for calibrating measurements using standard samples and spot sizes.

    Attributes:
        calibration (dict): Stores linear regression results for each isotope.
        analyte_mass (dict): Stores calculated analyte masses for each isotope.
        analyte_intensity (dict): Stores mean peak intensities for each isotope.

    Args:
        standards (list[tuple]): List of tuples, each containing indices or references to Measurement objects used as standards.
        spotsizes (list): List of spot sizes, either as tuples (x, y, z) or (d, z), where x, y, d are in micrometers and z in nanometers.
        film_thickness (list): List of film thickness values (nanometers) for each standard.
        film_concentration_percent (float): Concentration of analyte in the film, as a percentage.

    Methods:
        merge_peak_df(measurements, isotope):
            Merges the peak dataframes for a given isotope from a list of Measurement objects.

        reg_function(isotope, analyte_ppm_film=0, force_zero=False, mass_correction=1):
            Calculates analyte mass and intensity for calibration, performs linear regression, and stores the result.

        reg_plot_2(isotope):
            Plots the calibration curve for a given isotope using matplotlib, including the regression line and R² value.

        reg_plot(isotope, confidence_interval=90):
            Plots the calibration data and regression using seaborn with a specified confidence interval.
    """    

    def __init__(self, standards: list[tuple]):
        """
        Initializes the calibration object with standard data, spot sizes, film thicknesses, and film concentration.

        Args:
            standards (list[tuple]): A list of tuples representing the calibration standards.
            spotsizes (list): A list of spot sizes used in the calibration.
            film_thickness (list): A list of film thickness values.
            film_concentration_percent (float): The concentration of the film in percent.

        Attributes:
            calibration (dict): Dictionary to store calibration data.
            analyte_mass (dict): Dictionary to store analyte mass data.
            analyte_intensity (dict): Dictionary to store analyte intensity data.
        """
        self.__standards = standards
        self.calibration = dict()
        self.analyte_mass = dict()
        self.analyte_area = dict()

    def merge_peak_df(self, measurements: list, isotope: str):
        """
        Merges peak dataframes for a specified isotope from a list of measurement objects.

        Parameters:
            measurements (list): A list of measurement objects, each containing a 'peaks' attribute.
            isotope (str): The isotope key to extract the corresponding peak dataframe from each measurement.

        Returns:
            pandas.DataFrame: A concatenated dataframe containing all peak data for the specified isotope across the provided measurements.
        """
        df_list = []
        for m in measurements:
            df_list.append(m.peaks[isotope])
        df_merged = pd.concat(df_list, axis=0, ignore_index=True)
        return df_merged
    
    def ablated_mass_xy(self, isotope: str, x: list, y: list, z: list, polymer_percent: float, analyte_ppm_dry: float = 0, conversion = 1):
        m = []
        for x, y, z in zip(x, y, z):
            ablated_drymass = (z * 1e-7 * x * y * 1e-8) * (polymer_percent / 100)
            if analyte_ppm_dry != 0:
                ablated_drymass = ablated_drymass * analyte_ppm_dry * 1e-6
            m.append(ablated_drymass * conversion)
        self.analyte_mass[isotope] = m
        return m


    def ablated_mass_iva(self, isotope: str, d: list, z: list, polymer_percent: float, analyte_ppm_dry: float = 0, conversion = 1):
        z_mean = np.mean(z)
        z_err = np.std(z)
        diameters = d

        means = []
        errors = []
        for d in diameters:
            ablated_drymass = (z_mean * 1e-7 * (d/2)**2 * math.pi * 1e-8) * (polymer_percent / 100) 
            if analyte_ppm_dry != 0:
                ablated_drymass = ablated_drymass * analyte_ppm_dry * 1e-6
            ablated_drymass_err = ablated_drymass * (z_err / z_mean)
            means.append(ablated_drymass * conversion)
            errors.append(ablated_drymass_err * conversion)
        self.analyte_mass[isotope] = means
        return means, errors
    

    def mean_area(self, isotope: str):
        means = []
        errors = []
        for s in self.__standards:
            if type(s) == tuple:
                areas = pd.concat([m.peaks[isotope]['area'] for m in list(s)], ignore_index=True)
            else:
                areas = s.peaks[isotope]['area']
            means.append(areas.mean())
            errors.append(areas.std())
        self.analyte_area[isotope] = means
        return means, errors
    

    def regression(self, x, y, x_err, y_err, color, ci = 90, saveas = ""):
        x_fit = np.linspace(min(x), max(x), 100)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_fit = slope * x_fit + intercept

        fig, ax = plt.subplots(figsize=(7, 7 / 1.618))  # Golden ratio aspect ratio
        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err,
            fmt='o', label='data', markersize=5,
            elinewidth=1, capsize=3, capthick=1, color=color
        )
        ax.plot(
            x_fit, y_fit, linestyle='--', linewidth=1, color=color,
            label=f'fit (y = {slope:.2f} $\cdot$ x {"+" if intercept >= 0 else "-"} {abs(intercept):.2f}, $R^2$ = {r_value**2:.3f})'
        )

        ci = utils.conf_interval(x, y, y_err, slope, intercept, x_fit, alpha=(1-ci/100))
        ax.fill_between(x_fit, y_fit - ci, y_fit + ci, alpha=0.2, label='90\% confidence interval', color=color)

        ax.set_xlabel('mass (fg)', fontsize=14)
        ax.set_ylabel('peak area (a.u.)', fontsize=14)
        ax.legend(fontsize=12, frameon=False)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if saveas:
            ax.figure.savefig(saveas)
        plt.show()

        return slope, intercept, r_value, p_value, std_err



    def reg_function(self, isotope: str, analyte_ppm_film: float = 0, force_zero: bool = False, mass_correction: float = 1):
        """
        Calculates the calibration regression for a given isotope by relating ablated analyte mass to measured intensity.

        Parameters:
            isotope (str): The isotope for which the calibration is performed.
            analyte_ppm_film (float, optional): The analyte concentration in the film in ppm. If not provided, uses the default value of 0.
            force_zero (bool, optional): If True, forces the calibration curve through the origin by adding a zero point. Defaults to False.
            mass_correction (float, optional): Correction factor applied to the calculated ablated mass. Defaults to 1.

        Returns:
            scipy.stats._stats_mstats_common.LinregressResult: The result of the linear regression, containing slope, intercept, r-value, p-value, and standard error.

        Side Effects:
            Updates self.analyte_mass, self.analyte_intensity, and self.calibration for the given isotope.
            Prints the calibration equation and R^2 value to the console.
        """
        self.analyte_mass[isotope] = []
        self.analyte_intensity[isotope] = []

        if force_zero:
            self.analyte_mass[isotope].append(0)
            self.analyte_intensity[isotope].append(0)

        if type(self.__spotsizes[0]) == tuple:
            for i in range(len(self.__spotsizes)):
                x = self.__spotsizes[i][0]
                y = self.__spotsizes[i][1]
                z = self.__film_thickness[i]
                ablated_drymass = (z * 1e-7 * x * y * 1e-8) * (self.__film_concentration_percent / 100) * mass_correction
                if analyte_ppm_film != 0:
                    ablated_drymass = ablated_drymass * analyte_ppm_film * 1e-6
                self.analyte_mass[isotope].append(ablated_drymass)

        else:
            for i in range(len(self.__spotsizes)):
                d = self.__spotsizes[i]
                z = self.__film_thickness[i]
                ablated_drymass = (z * 1e-7 * (d/2)**2 * math.pi * 1e-8) * (self.__film_concentration_percent / 100) * mass_correction
                if analyte_ppm_film != 0:
                    ablated_drymass = ablated_drymass * analyte_ppm_film * 1e-6
                self.analyte_mass[isotope].append(ablated_drymass)

        for s in self.__standards:
            if type(s) == tuple:
                df = self.merge_peak_df(measurements = list(s), isotope = isotope)["area"].mean()
            else:
                df = s.peaks[isotope]["area"].mean()
            self.analyte_intensity[isotope].append(df)

        self.analyte_intensity[isotope] = [float(i) for i in self.analyte_intensity[isotope]]

        self.calibration[isotope] = sc.stats.linregress(x=self.analyte_mass[isotope], y=self.analyte_intensity[isotope])

        print(f'{isotope} calibration equation: y = {self.calibration[isotope].intercept:+.3e} {self.calibration[isotope].slope:+.3e} * x (R^2 = {(self.calibration[isotope].rvalue ** 2) * 100:.2f} %)')

        return self.calibration[isotope]


    def reg_plot(self, isotope: str):
        """
        Generate and display a scatter plot with a linear regression fit for the calibration data of a given isotope.

        This method plots the analyte mass (converted to femtograms) versus analyte intensity for the specified isotope.
        It fits a linear regression line to the data, displays the fit equation and R² value on the plot, and shows the plot.

        Parameters:
            isotope (str): The key corresponding to the isotope in the analyte_mass and analyte_intensity dictionaries.

        Returns:
            bool: Returns True after displaying the plot.

        Notes:
            - The x-axis represents the analyte mass in femtograms (fg).
            - The y-axis represents the peak area (arbitrary units).
            - The regression equation and R² value are shown in the plot legend.
        """

        x = np.array(self.analyte_mass[isotope]) * 1e15
        y = self.analyte_intensity[isotope]
        plt.figure()
        plt.scatter(x, y, label='Data')
        # Fit line
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        # Calculate R^2
        y_pred = slope * np.array(x) + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        eqn_label = (
            r'Fit: $y = %.2f + %.2fx$' '\n'
            r'$R^2 = %.3f$' % (intercept, slope, r2)
        )
        plt.plot(x_fit, y_fit, color='red', label=eqn_label)
        plt.title(f"{isotope} calibration")
        plt.xlabel(f'mass (fg)')
        plt.ylabel('peak area (a.u.)')
        plt.xlim([0, max(x) * 1.1])
        plt.ylim([0, max(y) * 1.1])
        plt.legend()
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()
        
        return True