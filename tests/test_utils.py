from pathlib import Path

import numpy as np

from tingan import utils
from tingan._tests.fake_data import fake_concatenated_file


def test_bin_min_max_standard() -> None:
    """Test that bin edges are correct."""
    array1 = [1, 2, 3, 4, 5]
    array2 = [6, 7, 8, 9]
    bin_edges = utils.bin_min_max((array1, array2))
    assert bin_edges[0] == array1[0]
    assert bin_edges[-1] == array2[-1]


def test_bin_min_max_with_nans() -> None:
    """Test that NaNs are exlucded from binning."""
    array1 = [1, 2, 3, np.nan, 5]
    array2 = [6, np.nan, 8, 9]
    bin_edges = utils.bin_min_max((array1, array2))
    assert bin_edges[0] == array1[0]
    assert bin_edges[-1] == array2[-1]


def test_gaussian_dist_basic() -> None:
    """Test that data statistics are correctly extracted."""
    data = 10 * [1]
    dist = utils.gaussian_dist(data)
    assert np.isclose(dist.rvs(), data[0])


def test_laplace_dist_basic() -> None:
    """Test that data statistics are correctly extracted."""
    data = 10 * [1]
    dist = utils.laplace_dist(data)
    assert np.isclose(dist.rvs(), data[0])


def test_split_file_at_string_nchunks() -> None:
    """Test that a file is split into the correct number of chunks."""
    path = fake_concatenated_file(3, "PSR_chunks")
    n = utils.split_file_at_string(path, "PSR")
    assert n == 3


def test_split_file_at_string_chunk_exists() -> None:
    """Test that chunked files are created."""
    path = fake_concatenated_file(2, "PSR_exists")
    utils.split_file_at_string(path, "PSR")
    sfx = path.suffix
    for i in range(2):
        assert Path(str(path)[: -len(sfx)] + f"_{i}" + sfx).exists()


def test_split_file_at_string_string_middle() -> None:
    """Test a file is split only when the string appears at the beginning of a line."""
    path = fake_concatenated_file(1, "random")
    n = utils.split_file_at_string(path, "random")
    assert n == 1
