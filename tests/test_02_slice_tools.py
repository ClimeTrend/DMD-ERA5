"""
Tests for the era5_processing module.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from dmd_era5 import (
    apply_delay_embedding,
    create_mock_era5,
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
    da = flatten_era5_variables(ds)
    assert da.values.ndim == 2, "Expected 2D data array"
    assert sorted(da.dims) == sorted(
        ["space", "time"]
    ), "Expected data dimensions to be space, time"
    assert sorted(da.coords) == sorted(
        ["space", "time", "original_variable"]
    ), "Expected coordinates to be space, time, original_variable"
    if mock_data == "mock_era5_temperature":
        assert sorted(da.attrs["original_variables"]) == [
            "temperature"
        ], "Expected variable to be temperature"
    elif mock_data == "mock_era5_temperature_wind":
        assert sorted(da.attrs["original_variables"]) == [
            "temperature",
            "u_component_of_wind",
        ], "Expected variables to be temperature and u_component_of_wind"


@pytest.mark.dependency(name="test_flatten_era5_variables_array_dims")
def test_flatten_era5_variables_array_dims(mock_era5_temperature_wind):
    """
    Test the dimensions of the flattened data array, and check that the data
    is correctly flattened.
    """
    da = flatten_era5_variables(mock_era5_temperature_wind)

    sizes = dict(mock_era5_temperature_wind.sizes)
    n_time = sizes["time"]
    n_lat = sizes["latitude"]
    n_lon = sizes["longitude"]
    n_level = sizes["level"]
    n_space = n_lat * n_lon * n_level
    n_vars = 2  # temperature and u_component_of_wind

    assert da.values.shape == (
        n_space * n_vars,
        n_time,
    ), "Expected flattened data shape to be (n_space * n_vars, n_time)"

    # index a few spatial locations to check that the data is correctly flattened
    lat_level_lon = [(1000, 40, 90), (1000, -20, -50), (850, 0, 0)]
    for coord in lat_level_lon:
        level, lat, lon = coord
        temperature = da.sel(space=(level, lat, lon))
        temperature = temperature.sel(
            space=temperature.original_variable == "temperature"
        ).values
        wind = da.sel(space=(level, lat, lon))
        wind = wind.sel(space=wind.original_variable == "u_component_of_wind").values
        assert np.allclose(
            temperature,
            mock_era5_temperature_wind.temperature.sel(
                level=level, latitude=lat, longitude=lon
            ).values,
        )
        assert np.allclose(
            wind,
            mock_era5_temperature_wind.u_component_of_wind.sel(
                level=level, latitude=lat, longitude=lon
            ).values,
        )


@pytest.mark.dependency(
    name="test_apply_delay_embedding_dataarray",
    depends=["test_flatten_era5_variables_array_dims"],
)
def test_apply_delay_embedding_dataarray(mock_era5_temperature_wind):
    """Test the apply_delay_embedding function with a DataArray."""

    da_flatten = flatten_era5_variables(mock_era5_temperature_wind)
    d = 2
    da_delay = apply_delay_embedding(da_flatten, d=d)
    assert isinstance(da_delay, xr.DataArray), "Expected output to be a DataArray"

    # Check the dimensions
    assert da_delay.ndim == 2, "Expected 2D data array"
    assert sorted(da_delay.dims) == sorted(
        ["space", "time"]
    ), "Expected data dimensions to be space, time"
    assert sorted(da_delay.coords) == sorted(
        ["space", "time", "original_variable", "delay"]
    ), "Expected coordinates to be space, time, original_variable, delay"
    assert da_delay.shape[0] == da_flatten.shape[0] * d, """
    Expected space dimension to be multiplied by delay.
    """
    assert da_delay.shape[1] == da_flatten.shape[1] - d + 1, """
    Expected time dimension to be reduced by delay-1.
    """

    # Check the coordinates
    n_space = da_flatten.sizes["space"]
    for coord in ["space", "original_variable"]:
        assert all(
            da_delay.coords[coord].values[:n_space] == da_flatten.coords[coord].values
        ), f"""
        Expected coordinate {coord} to be the same.
        """
        assert all(
            da_delay.coords[coord].values[n_space:] == da_flatten.coords[coord].values
        ), f"""
        Expected coordinate {coord} to be the same.
        """
    assert all(np.unique(da_delay.coords["delay"].values) == np.arange(d)), f"""
    Expected delay coordinate to be {np.arange(d)}.
    """
    assert all(
        da_delay.coords["time"].values == da_flatten.coords["time"].values[d - 1 :]
    ), """
    Expected time coordinate to be shifted by delay-1.
    """

    # Index a few spatial locations to check that the data is correctly delayed
    lat_level_lon = [(1000, 40, 90), (1000, -20, -50), (850, 0, 0)]
    for var in da_flatten.attrs["original_variables"]:
        for coord in lat_level_lon:
            level, lat, lon = coord
            variable_delay = da_delay.sel(space=(level, lat, lon))
            variable_delay = variable_delay.sel(
                space=variable_delay.original_variable == var
            )
            variable_delay_0 = np.squeeze(
                variable_delay.sel(space=variable_delay.delay == 0).values
            )
            variable_delay_1 = np.squeeze(
                variable_delay.sel(space=variable_delay.delay == 1).values
            )

            variable_flatten = da_flatten.sel(space=(level, lat, lon))
            variable_flatten = np.squeeze(
                variable_flatten.sel(
                    space=variable_flatten.original_variable == var
                ).values
            )

            assert np.allclose(
                variable_delay_0,
                variable_flatten[1:],
            ), """
            Expected delay 0 to be the same as the original data
            shifted by 1 time step
            """
            assert np.allclose(
                variable_delay_1,
                variable_flatten[:-1],
            ), """
            Expected delay 1 to be the same as the original data
            shortened by 1 time step
            """
