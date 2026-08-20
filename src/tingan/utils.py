"""Some useful functions for data manipulation and analysis."""

from pathlib import Path

import numpy as np
from scipy.stats import _continuous_distns, laplace, norm


def load_latex_table(texfile: str) -> tuple:
    """
    Load data table from file in LaTeX format.

    param texfile: path to latex table file
    return: parameter values and (min-max)
    """
    t = np.loadtxt(texfile, dtype=object, delimiter="&", skiprows=3)
    tshape = t.shape
    rows = tshape[0]
    cols = tshape[1]
    e = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            tij = (
                t[i, j]
                .replace(" ", "")
                .replace("\\", "")
                .replace("{", "")
                .replace("}", "")
                .replace("--", "-")
                .replace("$", "")
            )
            if "(" in tij and ")" in tij and ":" not in tij:
                if tij[0] != "(":
                    tij = tij.split("(")
                    eij = tij[1].split(")")[0]
                    t[i, j] = float(tij[0])
                    e[i, j, :] = eij
                else:
                    tij = tij[1:-1].split(",")
                    t[i, j] = np.nan
                    e[i, j, :] = tij
            elif "^" in tij and "_" in tij and ":" not in tij:
                tij = tij.split("^")
                t[i, j] = float(tij[0])
                tij = tij[1].split("_")
                e[i, j, :] = t[i, j] - np.array(tij[::-1], dtype=float)
            elif tij == "NA":
                t[i, j] = np.nan
            else:
                t[i, j] = tij

    return t, e


def bin_min_max(arrays: tuple, nbins: int = 10) -> np.ndarray:
    """
    Create a regular binning that spans a set of arrays.

    param arrays: list of arrays
    param nbins: number of bins

    return: bin edges
    """
    array = np.concatenate(arrays).flatten()
    return np.linspace(np.nanmin(array), np.nanmax(array), nbins + 1)


def gaussian_dist(data: list | np.ndarray) -> _continuous_distns:
    """
    Return a Gaussian distribution with same mean and standard deviation as data.

    :param data: data to mimic.
    """
    return norm(loc=np.mean(data), scale=np.std(data))


def laplace_dist(data: list | np.ndarray) -> _continuous_distns:
    """
    Return a Laplace distribution with same location and scale parameter as data.

    :param data: data to mimic.
    """
    return laplace(loc=np.mean(data), scale=np.std(data))


def split_file_at_string(file: Path, string: str) -> int:
    """
    Split a file at given string.

    Useful to reverse concatenation of several files that start with the same string.

    :param file: path to file
    :param string: string indicating the beginning of a chunk
    :return: number of chunk founds
    """
    lines_to_write: list[str] = []
    n_files = 0
    [*_, prefix, suffix] = str(file).split(".")
    with Path(file).open(mode="r") as rfile:
        while True:
            read_line = rfile.readline()
            if not read_line or string == read_line[: len(string)]:
                if len(lines_to_write) > 0:
                    with (
                        Path(f"{prefix}_{n_files}")
                        .with_suffix(f".{suffix}")
                        .open(mode="w") as wfile
                    ):
                        for line in lines_to_write:
                            wfile.write(line)
                    lines_to_write = []
                    n_files += 1
                if not read_line:
                    break
            lines_to_write.append(read_line)
    return n_files
