"""
Tests for the era5_processing module.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.slice_tools import (
    apply_delay_embedding,
    flatten_era5_variables,
    resample_era5_dataset,
    slice_era5_dataset,
    standardize_data,
)


@pytest.fixture
def mock_era5_temperature():
    """Create a mock ERA5 dataset with temperature data and multiple levels."""
    return create_mock_era5(
        start_datetime="2019-01-01T00:00",
        end_datetime="2019-01-05T00:00",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )


@pytest.fixture
def mock_era5_temperature_wind():
    """Create a mock ERA5 dataset with temperature and wind data and multiple levels."""
    return create_mock_era5(
        start_datetime="2019-01-01T00:00",
        end_datetime="2019-01-03T00:00",
        variables=["temperature", "u_component_of_wind"],
        levels=[1000, 850],
    )


def test_slice_era5_dataset(mock_era5_temperature):
    """Test that the slice_era5_dataset function correctly slices the dataset."""

    # Test slicing
    sliced_ds = slice_era5_dataset(
        mock_era5_temperature,
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


def test_resample_era5_dataset():
    """Test that the resample_era5_dataset function correctly resamples the dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000],
    )

    # Test resampling
    resampled_ds = resample_era5_dataset(mock_ds, timedelta(hours=6))

    assert len(resampled_ds.time) == 5, "Expected 5 time points"
    assert (
        resampled_ds.time.diff("time").astype("timedelta64[ns]").astype(int)
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


@pytest.mark.parametrize(
    ("X", "d", "expected"),
    [
        (np.array([[0, 1, 2, 3, 4]]), 1, np.array([[0, 1, 2, 3, 4]])),
        (np.array([[0, 1, 2, 3, 4]]), 2, np.array([[0, 1, 2, 3], [1, 2, 3, 4]])),
        (np.array([[0, 1, 2, 3, 4]]), 3, np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])),
        (
            np.array([[0, 1, 2], [3, 4, 5]]),
            2,
            np.array([[0, 1], [3, 4], [1, 2], [4, 5]]),
        ),
    ],
)
def test_apply_delay_embedding(X, d, expected):
    """Test the apply_delay_embedding function."""
    output = apply_delay_embedding(X, d)
    assert np.array_equal(output, expected), "Output matrix not as expected"


@pytest.mark.parametrize(
    "X",
    [
        (
            np.zeros(
                3,
            )
        ),
        (np.zeros((3, 3, 3))),
    ],
)
def test_apply_delay_embedding_invalid_matrix(X):
    """Test the apply_delay_embedding function with invalid input matrix."""
    with pytest.raises(ValueError, match="Input array must be 2D."):
        apply_delay_embedding(X, 1)


@pytest.mark.parametrize(
    "d",
    [
        (0),
        (0.5),
        (-1),
    ],
)
def test_apply_delay_embedding_invalid_delay(d):
    """Test the apply_delay_embedding function with invalid delay."""
    with pytest.raises(ValueError, match="Delay must be an integer greater than 0."):
        apply_delay_embedding(np.zeros((3, 3)), d)


@pytest.mark.parametrize(
    "mock_data", ["mock_era5_temperature", "mock_era5_temperature_wind"]
)
def test_flatten_era5_variables_basic(mock_data, request):
    """Basic test for the flatten_era5_variables function."""

    ds = request.getfixturevalue(mock_data)
    data_combined, flattened_coords, variables = flatten_era5_variables(ds)
    assert data_combined.ndim == 2, "Expected 2D data array"
    assert sorted(flattened_coords.keys()) == sorted(
        ["level", "latitude", "longitude", "time"]
    ), "Expected coordinates to include level, latitude, longitude, and time"
    if mock_data == "mock_era5_temperature":
        assert sorted(variables) == [
            "temperature"
        ], "Expected variable to be temperature"
    elif mock_data == "mock_era5_temperature_wind":
        assert sorted(variables) == [
            "temperature",
            "u_component_of_wind",
        ], "Expected variables to be temperature and u_component_of_wind"


def test_flatten_era5_variables_array_dims(mock_era5_temperature_wind):
    """Test the dimensions of the flattened data array."""
    data_combined, _, _ = flatten_era5_variables(mock_era5_temperature_wind)

    sizes = dict(mock_era5_temperature_wind.sizes)
    n_time = sizes["time"]
    n_lat = sizes["latitude"]
    n_lon = sizes["longitude"]
    n_level = sizes["level"]
    n_space = n_lat * n_lon * n_level
    n_vars = 2  # temperature and u_component_of_wind

    assert data_combined.shape == (
        n_space * n_vars,
        n_time,
    ), "Expected flattened data shape to be (n_space * n_vars, n_time)"
