"""
Tests for the era5_download module.
"""

from datetime import datetime, timedelta

import pytest
import xarray as xr

from dmd_era5.era5_download import (
    config_parser,
    download_era5_data,
)


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all",
        "levels": "1000",
    }


def test_config_parser_basic(base_config):
    parsed_config = config_parser(base_config)

    assert (
        parsed_config["source_path"] == base_config["source_path"]
    ), f"""source_path should be {base_config['source_path']}
    not {parsed_config['source_path']}"""
    assert parsed_config["start_datetime"] == datetime(
        2019, 1, 1
    ), f"""start_datetime should be {datetime(2019, 1, 1, 0, 0)}
    not {parsed_config['start_datetime']}"""
    assert parsed_config["end_datetime"] == datetime(
        2020, 1, 1
    ), f"""end_datetime should be {datetime(2019, 1, 2, 0, 0)}
    not {parsed_config['end_datetime']}"""
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
        parsed_config["save_name"] == "2019-01-01T00_2020-01-01T00_1y.nc"
    ), f"""save_name should be 2019-01-01T00_2020-01-01T00_1y.nc
    not {parsed_config['save_name']}"""


# ----- Test cases -----


# --- Missing field
@pytest.mark.parametrize(
    "field",
    [
        "source_path",
        "start_datetime",
        "end_datetime",
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


# --- Invalid datetime
@pytest.mark.parametrize(
    ("datetime_field", "invalid_datetime"),
    [
        ("start_datetime", "2019-02-31"),
        ("start_datetime", "2019-13-01"),
        ("start_datetime", "2019-01-01T25"),
    ],
)
def test_config_parser_invalid_datetime(base_config, datetime_field, invalid_datetime):
    """Test the invalid datetime error."""
    base_config[datetime_field] = invalid_datetime
    with pytest.raises(ValueError, match="Invalid start"):
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


def test_config_parser_generate_save_name(base_config):
    """Test that the save_name is correctly generated."""
    base_config["start_datetime"] = "2023-01-01"
    base_config["end_datetime"] = "2023-12-31"
    base_config["delta_time"] = "1d"

    parsed_config = config_parser(base_config)

    expected_save_name = "2023-01-01T00_2023-12-31T00_1d.nc"
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


def test_download_era5_data_mock_with_slicing_and_thinning(base_config):
    """
    Test the full pipeline of downloading, slicing, and
    thinning ERA5 data using a mock dataset.
    """
    base_config["start_datetime"] = "2019-01-01T06"
    base_config["end_datetime"] = "2019-01-05T18"
    base_config["delta_time"] = "6h"
    base_config["levels"] = "1000,500"
    parsed_config = config_parser(base_config)

    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert era5_data.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 1, 6)
    assert era5_data.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 5, 18)
    assert list(era5_data.level.values) == [1000, 500]
    assert (
        era5_data.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all()
