"""
Tests for the era5_processing module.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.slice_tools import slice_era5_dataset, standardize_data, thin_era5_dataset


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
    ) == datetime(2019, 1, 2, 6), "Expected start time to be 2019-01-02 06:00"
    assert sliced_ds.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 4, 23), "Expected end time to be 2019-01-04 23:00"
    assert list(sliced_ds.level.values) == [
        1000,
        500,
    ], "Expected levels to be [1000, 500]"


def test_slice_era5_dataset_invalid_time():
    """Test the invalid time range error in the slice_era5_dataset function."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01T00:00",
        end_datetime="2019-01-05T00:00",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    with pytest.raises(ValueError, match="Time range .* is outside dataset"):
        slice_era5_dataset(
            mock_ds,
            start_datetime="2018-12-31T00:00",
            end_datetime="2019-01-05T00:00",
        )


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

    assert len(thinned_ds.time) == 5, "Expected 5 time points"
    assert (
        thinned_ds.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all(), "Expected time delta to be 6 hours"


def test_standardize_data():
    """Test the standardize_data function."""
    mock_era5 = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-10",
        variables=["temperature"],
        levels=[1000],
    )
    temperature_standardized = standardize_data(mock_era5["temperature"])
    assert np.allclose(temperature_standardized.values.mean(), 0), "Expected mean 0"
    assert np.allclose(temperature_standardized.values.std(), 1), "Expected std 1"


def test_standardize_data_no_scale():
    """Test standardize_data with scale=False."""
    mock_era5 = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-10",
        variables=["temperature"],
        levels=[1000],
    )
    original_std = mock_era5["temperature"].std(dim="time", keepdims=True)
    temperature_standardized = standardize_data(mock_era5["temperature"], scale=False)
    assert np.allclose(
        temperature_standardized.mean(dim="time", keepdims=True), 0
    ), "Expected mean 0"
    assert np.allclose(
        temperature_standardized.std(dim="time", keepdims=True), original_std
    ), "Expected std to be unchanged"


def test_standardize_data_different_dimension():
    """Test standardize_data along a different dimension."""
    mock_era5 = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-10",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )
    temperature_standardized = standardize_data(mock_era5["temperature"], dim="level")
    assert np.allclose(
        temperature_standardized.mean(dim="level").values, 0
    ), "Expected mean 0"
    assert np.allclose(
        temperature_standardized.std(dim="level").values, 1
    ), "Expected std 1"
