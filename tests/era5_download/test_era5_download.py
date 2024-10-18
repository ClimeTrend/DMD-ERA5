"""
Tests for the era5_download module.
"""

from datetime import datetime, timedelta

import pytest
import xarray as xr

from dmd_era5.era5_download import (
    config_parser,
    create_mock_era5,
    download_era5_data,
    slice_era5_dataset,
    thin_era5_dataset,
)


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_date": "2019-01-01",
        "start_time": "00:00:00",
        "end_date": "2020-01-01",
        "end_time": "00:00:01",
        "delta_time": "1y",
        "variables": "all",
        "levels": "1000",
        "save_name": "",
    }


def test_config_parser_basic(base_config):
    parsed_config = config_parser(base_config)

    assert (
        parsed_config["source_path"] == base_config["source_path"]
    ), f"""source_path should be {base_config['source_path']}
    not {parsed_config['source_path']}"""
    assert parsed_config["start_date"] == datetime(
        2019, 1, 1
    ), f"""start_date should be {datetime(2019, 1, 1, 0, 0)}
    not {parsed_config['start_date']}"""
    assert parsed_config["end_date"] == datetime(
        2020, 1, 1
    ), f"""end_date should be {datetime(2019, 1, 2, 0, 0)}
    not {parsed_config['end_date']}"""
    assert parsed_config["delta_time"] == timedelta(
        days=365
    ), f"delta_time should be {timedelta(hours=1)} not {parsed_config['delta_time']}"
    assert parsed_config["variables"] == [
        "all"
    ], f"variables should be ['all'] not {parsed_config['variables']}"
    assert parsed_config["levels"] == [
        1000
    ], f"levels should be [1000] not {parsed_config['levels']}"
    assert (
        parsed_config["save_name"] == "2019-01-01_2020-01-01_1y.nc"
    ), f"""save_name should be 2019-01-01_2020-01-01_1y.nc
    not {parsed_config['save_name']}"""


# ----- Test cases -----


# --- Missing field
@pytest.mark.parametrize(
    "field",
    [
        "source_path",
        "start_date",
        "start_time",
        "end_date",
        "end_time",
        "delta_time",
        "variables",
        "levels",
    ],
)
def test_config_parser_missing_field(base_config, field):
    """Test the missing field error."""
    del base_config[field]
    with pytest.raises(ValueError, match=f"Missing required field in config: {field}"):
        config_parser(base_config)


# --- Invalid date
@pytest.mark.parametrize(
    ("date_field", "invalid_date"),
    [
        ("start_date", "2019-02-31"),
        ("start_date", "2019-13-01"),
        ("start_date", "not-a-date"),
    ],
)
def test_config_parser_invalid_date(base_config, date_field, invalid_date):
    """Test the invalid date error."""
    base_config[date_field] = invalid_date
    with pytest.raises(ValueError, match="Invalid start date or time format"):
        config_parser(base_config)


# --- Invalid time
@pytest.mark.parametrize(
    ("time_field", "invalid_time"),
    [
        ("start_time", "25:00:00"),
        ("start_time", "12:60:00"),
        ("start_time", "not-a-time"),
    ],
)
def test_config_parser_invalid_time(base_config, time_field, invalid_time):
    """Test the invalid time error."""
    base_config[time_field] = invalid_time
    with pytest.raises(ValueError, match="Invalid start date or time format"):
        config_parser(base_config)


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ("1000,850,500", [1000, 850, 500]),
        ("1000", [1000]),
        (" 1000 , 850 ", [1000, 850]),
    ],
)
def test_config_parser_levels(base_config, levels, expected):
    """Test the levels field."""
    base_config["levels"] = levels
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["levels"] == expected
    ), f"Expected levels to be {expected}, but got {parsed_config['levels']}"


@pytest.mark.parametrize(
    ("delta_time", "expected"),
    [
        ("1h", timedelta(hours=1)),
        ("24h", timedelta(hours=24)),
        ("1d", timedelta(days=1)),
        ("7d", timedelta(days=7)),
    ],
)
def test_config_parser_delta_time(base_config, delta_time, expected):
    """Test the delta_time field."""
    base_config["delta_time"] = delta_time
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["delta_time"] == expected
    ), f"Expected delta_time to be {expected}, but got {parsed_config['delta_time']}"


@pytest.mark.parametrize("invalid_delta", ["1v", "1", "not-a-delta"])
def test_config_parser_invalid_delta_time(base_config, invalid_delta):
    """Test the invalid delta_time error."""
    base_config["delta_time"] = invalid_delta
    with pytest.raises(ValueError, match="Error parsing delta_time"):
        config_parser(base_config)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ("temperature,humidity,pressure", ["temperature", "humidity", "pressure"]),
        ("all", ["all"]),
        (" temperature , humidity ", ["temperature", "humidity"]),
    ],
)
def test_config_parser_variables(base_config, variables, expected):
    """Test the variables field."""
    base_config["variables"] = variables
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["variables"] == expected
    ), f"Expected variables to be {expected}, but got {parsed_config['variables']}"


# --- Custom save_name
@pytest.mark.parametrize(
    "save_name", ["custom_name.nc", "era5_data.nc", "ERA5_data.nc"]
)
def test_config_parser_custom_save_name(base_config, save_name):
    """Test the custom save_name."""
    base_config["save_name"] = save_name
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["save_name"] == save_name
    ), f"Expected save_name to be {save_name}, but got {parsed_config['save_name']}"


def test_config_parser_generate_save_name(base_config):
    """Test that the save_name is correctly generated when left blank."""
    base_config["save_name"] = ""  # Set save_name to an empty string
    base_config["start_date"] = "2023-01-01"
    base_config["end_date"] = "2023-12-31"
    base_config["delta_time"] = "1d"

    parsed_config = config_parser(base_config)

    expected_save_name = "2023-01-01_2023-12-31_1d.nc"
    assert (
        parsed_config["save_name"] == expected_save_name
    ), f"""Expected save_name to be {expected_save_name},
    but got {parsed_config['save_name']}"""


# ---- Test mock data ----


def test_download_era5_data_mock(base_config):
    """Test that the download_era5_data function correctly creates a mock dataset."""
    parsed_config = config_parser(base_config)

    # Use the mock dataset
    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert isinstance(era5_data, xr.Dataset), "The result should be an xarray Dataset"
    assert (
        "temperature" in era5_data.variables
    ), "The dataset should contain temperature data"
    assert (
        era5_data.attrs["source"] == "Generated mock data"
    ), "The dataset should have the mock data source attribute"


def test_slice_era5_dataset():
    """Test that the slice_era5_dataset function correctly slices the dataset."""
    mock_ds = create_mock_era5(
        start_date="2019-01-01",
        end_date="2019-01-05",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    # Test slicing
    sliced_ds = slice_era5_dataset(
        mock_ds,
        start_date=datetime(2019, 1, 2),
        end_date=datetime(2019, 1, 4),
        levels=[1000, 500],
    )

    assert sliced_ds.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 2)
    assert sliced_ds.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 4)
    assert list(sliced_ds.level.values) == [1000, 500]


def test_thin_era5_dataset():
    """Test that the thin_era5_dataset function correctly thins the dataset."""
    mock_ds = create_mock_era5(
        start_date="2019-01-01",
        end_date="2019-01-02",
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


def test_download_era5_data_mock_with_slicing_and_thinning(base_config):
    """
    Test the full pipeline of downloading, slicing, and
    thinning ERA5 data using a mock dataset.
    """
    base_config["start_date"] = "2019-01-01"
    base_config["end_date"] = "2019-01-05"
    base_config["delta_time"] = "6h"
    base_config["levels"] = "1000,500"
    parsed_config = config_parser(base_config)

    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert era5_data.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 1)
    assert era5_data.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 5)
    assert list(era5_data.level.values) == [1000, 500]
    assert (
        era5_data.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all()
