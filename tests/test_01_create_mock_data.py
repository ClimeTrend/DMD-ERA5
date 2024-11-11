"""
Tests for the create_mock_data module.
"""

import xarray as xr

from dmd_era5.create_mock_data import create_mock_era5


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
