"""
Tests for the create_mock_data module.
"""

import numpy as np
import xarray as xr

from dmd_era5 import create_mock_era5, create_mock_era5_svd


def test_create_mock_era5():
    """Test that the create_mock_era5 function correctly creates a mock ERA5 dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    assert isinstance(
        mock_ds, xr.Dataset
    ), "Expected create_mock_era5 to return an xarray Dataset."
    assert (
        "temperature" in mock_ds.data_vars
    ), "Expected 'temperature' variable in mock ERA5 dataset."
    assert list(mock_ds.level.values) == [
        1000,
        850,
        500,
    ], "Expected levels to be [1000, 850, 500]."


def test_create_mock_era5_svd():
    """
    Test that the create_mock_era5_svd function correctly creates
    mock ERA5 SVD results.
    """
    U, s, V, coords, X = create_mock_era5_svd(n_components=4)

    assert isinstance(U, np.ndarray), "Expected U to be a numpy array."
    assert isinstance(s, np.ndarray), "Expected s to be a numpy array."
    assert isinstance(V, np.ndarray), "Expected V to be a numpy array."
    assert isinstance(
        coords, xr.Coordinates
    ), "Expected coords to be an xarray Coordinates."
    assert isinstance(X, xr.DataArray), "Expected X to be an xarray DataArray."
    assert U.shape[1] == 4, "Expected U to have 4 columns."
    assert s.size == 4, "Expected s to have 4 singular values."
    assert V.shape[0] == 4, "Expected V to have 4 rows."
    assert sorted(coords.keys()) == sorted(
        ["space", "time", "original_variable", "delay"]
    ), """
    Expected coords to have keys ['space', 'time', 'original_variable', 'delay'].
    """
