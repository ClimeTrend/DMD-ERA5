"""
Tests for the era5_processing module.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from dmd_era5.slice_tools import slice_era5_dataset, thin_era5_dataset, standardize_data
from dmd_era5.create_mock_data import create_mock_era5



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


def test_slice_era5_dataset_invalid_time():
    """Test the invalid time range error in the slice_era5_dataset function."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01T00:00",
        end_datetime="2019-01-05T00:00",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    with pytest.raises(ValueError, match="Requested time range .* is outside dataset bounds"):
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

    assert len(thinned_ds.time) == 5  # 24 hours / 6 hour intervals + 1
    # Check if 6 hours with some tolerance
    assert (
        thinned_ds.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all()




def test_standardize_data():
    """Test the standardize_data function."""
    mock_era5 = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-10",
        variables=["temperature"],
        levels=[1000],
    )
    temperature_standardized = standardize_data(mock_era5["temperature"])
    assert np.allclose(temperature_standardized.values.mean(), 0)
    assert np.allclose(temperature_standardized.values.std(), 1)

# TODO: these tests are not working

# def test_standardize_data_no_mean_center():
#     """Test standardize_data with mean_center=False."""
#     mock_era5 = create_mock_era5(
#         start_datetime="2019-01-01",
#         end_datetime="2019-01-10",
#         variables=["temperature"],
#         levels=[1000],
#     )
#     original_mean = mock_era5["temperature"].mean(dim="time", keepdims=True)
    
#     # Ensure no NaNs in the mock data
#     assert not np.isnan(mock_era5["temperature"]).any(), "Mock data contains NaNs"

#     temperature_standardized = standardize_data(
#         mock_era5["temperature"], 
#         mean_center=False, 
#         scale=True
#     )
    
#     # Print intermediate values for debugging
#     print("Original Mean:", original_mean.values)
#     print("Standardized Mean:", temperature_standardized.mean(dim="time", keepdims=True).values)
#     print("Standardized Std:", temperature_standardized.std(dim="time", keepdims=True).values)

#     # Mean should be unchanged along time dimension
#     assert np.allclose(
#         temperature_standardized.mean(dim="time", keepdims=True), 
#         original_mean,
#         atol=1e-7  # Adjust tolerance if needed
#     )
#     # Std should be 1 along time dimension
#     assert np.allclose(
#         temperature_standardized.std(dim="time", keepdims=True), 
#         1.0,
#         atol=1e-7  # Adjust tolerance if needed
#     )

# def test_standardize_data_no_scale():
#     """Test standardize_data with scale=False."""
#     mock_era5 = create_mock_era5(
#         start_datetime="2019-01-01",
#         end_datetime="2019-01-10",
#         variables=["temperature"],
#         levels=[1000],
#     )
#     original_std = mock_era5["temperature"].std(dim="time", keepdims=True)
#     temperature_standardized = standardize_data(
#         mock_era5["temperature"], 
#         mean_center=True, 
#         scale=False
#     )
#     # Mean should be 0 along time dimension
#     assert np.allclose(
#         temperature_standardized.mean(dim="time", keepdims=True), 
#         0
#     )
#     # Std should be unchanged along time dimension
#     assert np.allclose(
#         temperature_standardized.std(dim="time", keepdims=True), 
#         original_std
#     )

# def test_standardize_data_different_dimension():
#     """Test standardize_data along a different dimension."""
#     mock_era5 = create_mock_era5(
#         start_datetime="2019-01-01",
#         end_datetime="2019-01-10",
#         variables=["temperature"],
#         levels=[1000, 850, 500],
#     )
#     temperature_standardized = standardize_data(
#         mock_era5["temperature"], 
#         dim="level"
#     )
#     # Check mean and std along the level dimension
#     assert np.allclose(temperature_standardized.mean(dim="level").values, 0)
#     assert np.allclose(temperature_standardized.std(dim="level").values, 1)

# def test_standardize_data_no_modifications():
#     """Test standardize_data with both mean_center and scale set to False."""
#     mock_era5 = create_mock_era5(
#         start_datetime="2019-01-01",
#         end_datetime="2019-01-10",
#         variables=["temperature"],
#         levels=[1000],
#     )
#     original_data = mock_era5["temperature"].copy()
#     temperature_standardized = standardize_data(
#         mock_era5["temperature"], 
#         mean_center=False, 
#         scale=False
#     )
#     # Data should be unchanged
#     assert np.allclose(temperature_standardized.values, original_data.values)
