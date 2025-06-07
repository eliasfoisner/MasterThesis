import pandas as pd
import numpy as np
import warnings


class GDCParticles:
    """
    GDCParticles is a class for evaluating and analyzing paired Ce and Gd isotope peaks from measurement data, specifically for Gadolinium-doped Ceria (GDC) particles.

    This class provides methods to:
    - Pair Ce and Gd isotope peaks within a specified time window.
    - Calculate constituent masses for Ce and Gd using calibration data.
    - Compute the total particle mass, including oxygen, based on stoichiometry.
    - Estimate particle mass separately from Ce and Gd, assuming fixed molar composition.
    - Calculate particle diameter from mass and density.

    Attributes:
        particles (pd.DataFrame): DataFrame containing paired Ce and Gd peaks and calculated properties.

    Methods:
        __init__(measurement: object, ce_isotope: str = "CeO/156", gd_isotope: str = "GdO/174", time_window: float = 0.1)
            Initializes the GDCParticles object by pairing Ce and Gd peaks within a specified time window.

        constituent_mass(calibration: object)
            Calculates and appends the constituent masses ('ce_mass' and 'gd_mass') to the particles dataframe using calibration data.

        particle_mass(mw_ce: float, mw_gd: float)
            Calculates the total mass of GDC particles, including Ce, Gd, and O, based on stoichiometry.

        particle_mass_separately(mw_ce: float, mw_gd: float, assumed_ce: float, assumed_gd: float, assumed_ox: float)
            Computes the GDC particle mass separately using the masses of Ce and Gd, assuming a fixed molar composition.
            
        particle_diameter(mass: pd.DataFrame, density: float = 7.32)
            Calculates the diameter of GDC particles based on their mass and density.
    """

    def __init__(self, measurement: object, ce_isotope: str = "CeO/156", gd_isotope: str = "GdO/174", time_window: float = 0.001):
        """
        Initializes the evaluation object by pairing Ce and Gd isotope peaks within a specified time window.
        Args:
            measurement (object): The measurement object containing peak data for isotopes.
            ce_isotope (str, optional): The key for the Ce isotope in the measurement peaks. Defaults to "CeO/156".
            gd_isotope (str, optional): The key for the Gd isotope in the measurement peaks. Defaults to "GdO/174".
            time_window (float, optional): The maximum allowed time difference (in the same units as peak times) to consider Ce and Gd peaks as a pair. Defaults to 0.1.
        Attributes:
            particles (pd.DataFrame): A DataFrame containing paired Ce and Gd peaks, with columns:
                - "ce_time": Time of the Ce peak.
                - "ce_area": Area of the Ce peak.
                - "gd_time": Time of the Gd peak.
                - "gd_area": Area of the Gd peak.
        Notes:
            Only pairs where both Ce and Gd peaks are found within the specified time window are included.
        """
        self.__ce_isotope = ce_isotope
        self.__gd_isotope = gd_isotope

        ce_df = measurement.peaks[ce_isotope][["time", "area"]].copy()
        ce_df["isotope"] = ce_isotope
        gd_df = measurement.peaks[gd_isotope][["time", "area"]].copy()
        gd_df["isotope"] = gd_isotope

        ce_df = ce_df.reset_index(drop=True)
        gd_df = gd_df.reset_index(drop=True)

        # we need to ensure that only entries are copied where both Ce and Gd are found:
        pairs = []
        for _, ce_row in ce_df.iterrows():
            gd_candidates = gd_df[np.abs(gd_df["time"] - ce_row["time"]) <= time_window]
            
            if not gd_candidates.empty:
                gd_row = gd_candidates.iloc[(np.abs(gd_candidates["time"] - ce_row["time"])).argmin()]
                if not pd.isna(ce_row["time"]) and not pd.isna(gd_row["time"]):
                    pairs.append({
                        "ce_time": ce_row["time"],
                        "ce_area": ce_row["area"],
                        "gd_time": gd_row["time"],
                        "gd_area": gd_row["area"]
                    })
        pairs = [p for p in pairs if all(pd.notna([p["ce_time"], p["gd_time"]]))]
        self.particles = pd.DataFrame(pairs)


    def constituent_mass(self, slope_ce, intercept_ce, slope_gd, intercept_gd):
        """
        Calculates and appends the constituent masses ('ce_mass' and 'gd_mass') to the particles dataframe using a provided Calibration object.

        Parameters:
            calibration (object): An object containing calibration data for the relevant isotopes. It must provide a 'calibration' attribute, which is a mapping from isotope names to calibration parameters ('intercept' and 'slope').

        Returns:
            tuple: A tuple containing two pandas Series: (ce_mass, gd_mass), representing the calculated masses for the Ce and Gd isotopes, respectively.

        Side Effects:
            Modifies the 'particles' dataframe in-place by adding 'ce_mass' and 'gd_mass' columns.

        Raises:
            KeyError: If the required isotope keys are not present in the calibration object.
            AttributeError: If the calibration object or its entries do not have the expected attributes.
        """
        y_ce = self.particles['ce_area']
        d_ce = intercept_ce
        k_ce = slope_ce
        ce_mass = (y_ce - d_ce) / k_ce
        self.particles['ce_mass'] = ce_mass

        y_gd = self.particles['gd_area']
        d_gd = intercept_gd
        k_gd = slope_gd
        gd_mass = (y_gd - d_gd) / k_gd
        self.particles['gd_mass'] = gd_mass

        return ce_mass, gd_mass
    
    
    def particle_mass(self, mw_ce, mw_gd):
        """
        Calculates the total mass of GDC (Gadolinium-doped Ceria) particles, including contributions from cerium, gadolinium, and oxygen.

        Parameters:
            mw_ce (float): Molar mass of cerium (Ce).
            mw_gd (float): Molar mass of gadolinium (Gd).

        Returns:
            float: Total mass of the GDC particle, including Ce, Gd, and O components.

        Notes:
            - The calculation considers the stoichiometry of the GDC formula, including the oxygen content.
            - The method updates the 'particle_mass' entry in the self.particles dictionary.
        """    
        mw_o = 16.0
        m_ce = self.particles['ce_mass']
        m_gd = self.particles['gd_mass']
        n_ce = m_ce / mw_ce
        n_gd = m_gd / mw_gd
        x = n_gd / (n_ce + n_gd)
        n_ce_formula = 1 - x
        n_gd_formula = x
        n_o_formula = 2 * (n_ce_formula + n_gd_formula) - 0.5 * x
        o_ratio = n_o_formula / (n_ce_formula + n_gd_formula)
        n_o = (n_ce + n_gd) * o_ratio
        m_o = n_o * mw_o
        self.particles['particle_mass'] = m_ce + m_gd + m_o
        return m_ce + m_gd + m_o
    
    
    def particle_mass_separately(self, mw_ce, mw_gd, assumed_ce, assumed_gd, assumed_ox):
        """
        Computes the GDC (Gadolinium-doped Ceria) particle mass separately using the masses of Ce and Gd, 
        assuming a fixed molar composition of GDC.

        For each element (Ce and Gd), calculates a gravimetric factor based on the assumed molar fractions 
        and molar weights, then estimates the total particle mass by dividing the measured isotope mass by 
        the corresponding gravimetric factor.

        Parameters:
            mw_ce (float): Molar weight of cerium isotope (g/mol).
            mw_gd (float): Molar weight of gadolinium isotope (g/mol).
            assumed_ce (float): Assumed molar fraction of cerium in the GDC particle.
            assumed_gd (float): Assumed molar fraction of gadolinium in the GDC particle.
            assumed_ox (float): Assumed molar fraction of oxygen in the GDC particle.

        Returns:
            Tuple(pd.Series, pd.Series): 
                - First element: Particle mass calculated via cerium (pd.Series).
                - Second element: Particle mass calculated via gadolinium (pd.Series).
        """
        ce_mass = self.particles["ce_mass"]
        gd_mass = self.particles["gd_mass"]
        assumed_mass = assumed_ce * mw_ce + assumed_gd * mw_gd + assumed_ox * 16
        ce_gravfac = assumed_ce * mw_ce / assumed_mass
        gd_gravfac = assumed_gd * mw_gd / assumed_mass
        particle_mass_ce = ce_mass / ce_gravfac
        particle_mass_gd = gd_mass / gd_gravfac
        return particle_mass_ce, particle_mass_gd


    def particle_diameter(self, mass: pd.DataFrame, density: float = 7.09):
        """
        Calculates the diameter of GDC particles based on their mass and density.

        Parameters:
            mass (pd.DataFrame): DataFrame containing the mass of the particles.
            density (float, optional): Density of the GDC particles in g/cm³. Default is 7.32.

        Returns:
            pd.Series: Returns the computed particle diameters (in nm) as a pandas Series and stores them in self.particles['particle_diameter'].
        """
        m = mass
        rho = density
        d = (6 * m / (rho * np.pi)) ** (1/3) * 1e7
        self.particles['particle_diameter'] = d
        return d
    

    def molar_ratio(self):
        """
        Calculates the mean and standard deviation of the molar ratio of Gd2O3 to CeO2 in the particle dataset.

        The molar ratio is computed as:
            (gd_mass / (2 * 158)) / (ce_mass / 140)
        where:
            - ce_mass: Mass of Ce in each particle (assumed in grams).
            - gd_mass: Mass of Gd in each particle (assumed in grams).
            - 140: Molar mass of CeO2 (g/mol).
            - 158: Molar mass of Gd2O3 (g/mol), multiplied by 2 for stoichiometry.

        Returns:
            tuple: (mean_molar_ratio, std_molar_ratio)
                - mean_molar_ratio (float): Mean of the molar ratios across all particles.
                - std_molar_ratio (float): Standard deviation of the molar ratios across all particles.
        """
        ce_mass = self.particles["ce_mass"]
        gd_mass = self.particles["gd_mass"]
        ceo_mole = ce_mass / 140
        gdo_mole = gd_mass / (2 * 158)
        molar_ratio = gdo_mole / ceo_mole
        self.particles['molar_ratio'] = molar_ratio
        return molar_ratio