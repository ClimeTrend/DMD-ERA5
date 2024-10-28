"""
Tests for the utils module.
"""

from datetime import datetime, timedelta

import xarray as xr

from dmd_era5.utils import create_mock_era5, slice_era5_dataset, thin_era5_dataset


def test_create_mock_era5():
    """Test that the create_mock_era5 function correctly creates a mock ERA5 dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    assert isinstance(mock_ds, xr.Dataset)
    assert "temperature" in mock_ds.data_vars
    assert list(mock_ds.level.values) == [1000, 850, 500]


def test_slice_era5_dataset():
    """Test that the slice_era5_dataset function correctly slices the dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01T00:00",
        end_datetime="2019-01-05T00:00",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    # Test slicing
    sliced_ds = slice_era5_dataset(
        mock_ds,
        start_datetime="2019-01-02T06:00",
        end_datetime="2019-01-04T23:00",
        levels=[1000, 500],
    )

    assert sliced_ds.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 2, 6)
    assert sliced_ds.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 4, 23)
    assert list(sliced_ds.level.values) == [1000, 500]


def test_thin_era5_dataset():
    """Test that the thin_era5_dataset function correctly thins the dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000],
    )

    # Test thinning
    thinned_ds = thin_era5_dataset(mock_ds, timedelta(hours=6))

    assert len(thinned_ds.time) == 5  # 24 hours / 6 hour intervals + 1
    # Check if 6 hours with some tolerance
    assert (
        thinned_ds.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all()
