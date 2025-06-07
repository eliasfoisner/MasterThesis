# SinglePulse Library

**SinglePulse** is a Python library for processing, analyzing, and visualizing single-pulse ICP-MS data, with a focus on Perkin Elmer Syngistix and Nanomodul files. It provides tools for peak detection, background correction, calibration, particle analysis, and more.

**Please do not edit any code in the main branch of this project! You are free to create a private fork of this project and edit the code to your needs.**

## Features

- Syngistix & Nanomodul file support
- Peak detection and integration
- Background correction (global and local)
- Calibration and regression plotting
- Particle evaluation (e.g., GDC particles)
- Interactive and static plotting utilities

## Usage

To use this package, please install all requirements to your virtual environment using the following command:
```bash
pip install -r requirements.txt
```
For an example workflow using this package, see example_notebook.ipynb.


# Documentation

This document lists all classes and functions in the `singlepulse` package, organized by module.

---

## `singlepulse.calibration`

### Classes
- **SyngistixCalibration**
  - `__init__(standards: list[tuple])`
  - `merge_peak_df(measurements: list, isotope: str)`
  - `ablated_mass_xy(isotope: str, x: list, y: list, z: list, polymer_percent: float, analyte_ppm_dry: float = 0, conversion = 1)`
  - `ablated_mass_iva(isotope: str, d: list, z: list, polymer_percent: float, analyte_ppm_dry: float = 0, conversion = 1)`
  - `mean_area(isotope: str)`
  - `regression(x, y, x_err, y_err, color, ci = 90, saveas = "")`
  - `reg_function(isotope: str, analyte_ppm_film: float = 0, force_zero: bool = False, mass_correction: float = 1)`
  - `reg_plot(isotope: str)`
  - `reg_plot_2(isotope)`

---

## `singlepulse.evaluation`

### Classes
- **GDCParticles**
  - `__init__(measurement: object, ce_isotope: str = "CeO/156", gd_isotope: str = "GdO/174", time_window: float = 0.1)`
  - `constituent_mass(slope_ce, intercept_ce, slope_gd, intercept_gd)`
  - `particle_mass(mw_ce, mw_gd)`
  - `particle_mass_separately(mw_ce, mw_gd, assumed_ce, assumed_gd, assumed_ox)`
  - `particle_diameter(mass: pd.DataFrame, density: float = 7.32)`
  - `molar_ratio()`

---

## `singlepulse.measurement`

### Classes
- **SyngistixMeasurement**
  - `__init__(file_path: str)`
  - `timescale(isotope: str, cycle_time: float = False)`
  - `savgol(isotope: str, window_length: int = 100, polyorder: int = 2, deriv: int = 0)`
  - `peak_finding(isotope: str, threshold: float, distance: float)`
  - `peak_width(isotope: str, criterion: int)`
  - `peak_background(isotope: str)`
  - `peak_area(isotope: str, resize: int)`
  - `calibrate(isotope, calibration: object)`
  - `area_ratio(isotope_one: str, gravfac_one: float, isotope_two: str, gravfac_two: float)`
  - `plot(isotope: str, fig: object, savgol: bool = False, integration: bool = False, peaks: bool = False, background: bool = False, width: bool = False)`

---

## `singlepulse.utils`

### Functions
- `extract_string(text: str, first_character: str, second_character: str, right: bool = True)`
- `generate_colors(n, colormap_name: str = 'viridis')`
- `generate_marker_list(n, available_markers)`
- `check_nanomodul(file)`
- `import_folder(path)`
- `conf_interval(x, y, yerr, slope, intercept, x_fit, alpha=0.10)`

---