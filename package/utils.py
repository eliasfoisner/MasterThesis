import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, t
import numpy as np


def extract_string(text: str, first_character: str, second_character: str, right: bool = True):
    """
    Extracts a substring from the input text that is located between two specified characters or substrings.
    Args:
        text (str): The input string from which to extract the substring.
        first_character (str): The character or substring marking one boundary.
        second_character (str): The character or substring marking the other boundary.
        right (bool, optional): If True, extracts the substring to the right of the first occurrence of `first_character` up to the next occurrence of `second_character`.
            If False, extracts the substring to the left of the first occurrence of `first_character` up to the previous occurrence of `second_character`. Defaults to True.
    Returns:
        str or None: The extracted substring if both boundaries are found; otherwise, None.
    """
    if right:
        first_index = text.find(first_character)
        second_index = text.find(second_character, first_index+len(first_character), len(text))
        if first_index != -1 and second_index != -1:
            value = text[first_index+len(first_character):second_index]
            return value
        return None
    else:
        first_index = text.find(first_character)
        second_index = text.rfind(second_character, 0, first_index)
        if first_index != -1 and second_index != -1:
            value = text[second_index+len(second_character):first_index]
            return value
        return None


def generate_colors(n, colormap_name: str = 'viridis'):
    """
    Generate a list of distinct colors from a specified matplotlib colormap.
    Args:
        n (int): The number of colors to generate.
        colormap_name (str, optional): The name of the matplotlib colormap to use. Defaults to 'viridis'.
    Returns:
        list: A list of RGBA color tuples sampled from the specified colormap.
    Example:
        colors = generate_colors(5, colormap_name='plasma')
    """
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i/n) for i in range(n)]
    return colors


def generate_marker_list(n, available_markers):
    """
    Generates a list of markers of length `n` by cycling through the provided list of available markers.
    If `n` is greater than the number of available markers, the markers are repeated in order until the list reaches length `n`.
    Args:
        n (int): The number of markers to generate.
        available_markers (list): A list of marker values to use.
    Returns:
        list: A list of markers of length `n`, cycling through `available_markers` as needed.
    """
    markers = [available_markers[i % len(available_markers)] for i in range(n)]
    return markers


def check_nanomodul(file):
    """
    Checks if the first element in the first column of a CSV file is not a string.
    Parameters:
        file (str or file-like object): Path to the CSV file or a file-like object to read.
    Returns:
        bool: True if the first element is not a string (indicating a "nanomodul" file), False otherwise.
    """
    data = pd.read_csv(file, skiprows=0)
    if type(data.iloc[0, 0]) == str:
        return False
    else:
        return True
    

def import_folder(path):
    """
    Imports and lists all CSV files in the specified folder, sorted by modification time (most recent first).
    Args:
        path (str or Path): The path to the folder containing CSV files.
    Returns:
        list[Path]: A list of Path objects representing the CSV files found, sorted by modification time (descending).
    Prints:
        The names of the found CSV files with their corresponding indices.
    """
    dir = Path(path)
    files = sorted([f for f in dir.glob("*.csv")], key=lambda f: f.stat().st_mtime, reverse=True)
    print(f"Following files were found in folder '{dir.name}':\n")
    for i, f in enumerate(files):
        print(i, f.name)
    return files


def conf_interval(x, y, yerr, slope, intercept, x_fit, alpha=0.10):
    n = len(x)
    dof = n - 2
    tval = t.ppf(1 - alpha/2, dof)
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    y_pred = slope * x + intercept
    s_err = np.sqrt(np.sum(((y - y_pred) ** 2)) / dof)
    mean_x = np.mean(x)
    confs = []
    for xf in x_fit:
        se = s_err * np.sqrt(1/n + (xf - mean_x)**2 / np.sum((x - mean_x)**2))
        confs.append(tval * se)
    return np.array(confs)