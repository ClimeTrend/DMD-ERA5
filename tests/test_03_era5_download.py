"""
Tests for the era5_download module.
"""

import os
from datetime import datetime, timedelta

import pytest
import xarray as xr
from pyprojroot import here

from dmd_era5 import config_parser
from dmd_era5.constants import ERA5_PRESSURE_LEVEL_VARIABLES, ERA5_PRESSURE_LEVELS
from dmd_era5.era5_download import download_era5_data


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all_pressure_level_vars",
        "levels": "1000",
    }


def test_config_parser_basic(base_config):
    parsed_config = config_parser(base_config, section="era5-download")

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
    assert parsed_config["variables"] == list(ERA5_PRESSURE_LEVEL_VARIABLES), f"""
    variables should be {list(ERA5_PRESSURE_LEVEL_VARIABLES)}
    not {parsed_config['variables']}
    """
    assert parsed_config["levels"] == [
        1000
    ], f"levels should be [1000] not {parsed_config['levels']}"
    assert (
        parsed_config["save_name"] == "2019-01-01T00_2020-01-01T00_1y.nc"
    ), f"""save_name should be 2019-01-01T00_2020-01-01T00_1y.nc
    not {parsed_config['save_name']}"""
    assert parsed_config["save_path"] == os.path.join(
        here(), "data", "era5_download", parsed_config["save_name"]
    ), f"""
    save_path should be
    {os.path.join(here(), 'data', 'era5_download', parsed_config['save_name'])}
    not {parsed_config['save_path']}
    """
    assert parsed_config["era5_slice_path"] == os.path.join(
        here(), "data", "era5_download", parsed_config["save_name"]
    ), f"""
    era5_slice_path should be
    {os.path.join(here(), 'data', 'era5_download', parsed_config['save_name'])}
    not {parsed_config['era5_slice_path']}
    """


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
        config_parser(base_config, section="era5-download")


# --- Test the level field
@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ("1000,850,500", [1000, 850, 500]),
        ("1000", [1000]),
        (" 1000 , 850 ", [1000, 850]),
        ("all", list(ERA5_PRESSURE_LEVELS)),
    ],
)
def test_config_parser_levels(base_config, levels, expected):
    """Test the levels field."""
    base_config["levels"] = levels
    parsed_config = config_parser(base_config, section="era5-download")
    assert (
        parsed_config["levels"] == expected
    ), f"Expected levels to be {expected}, but got {parsed_config['levels']}"


# Test the case where the levels are not valid
def test_config_parser_levels_invalid(base_config):
    """Test the levels field with an invalid value."""
    base_config["levels"] = "965"
    with pytest.raises(ValueError, match="Unsupported level"):
        config_parser(base_config, section="era5-download")


# --- Test the variables field
@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        (
            "temperature,u_component_of_wind,v_component_of_wind",
            ["temperature", "u_component_of_wind", "v_component_of_wind"],
        ),
        ("all_pressure_level_vars", list(ERA5_PRESSURE_LEVEL_VARIABLES)),
        (" temperature , u_component_of_wind ", ["temperature", "u_component_of_wind"]),
    ],
)
def test_config_parser_variables(base_config, variables, expected):
    """Test the variables field."""
    base_config["variables"] = variables
    parsed_config = config_parser(base_config, section="era5-download")
    assert (
        parsed_config["variables"] == expected
    ), f"Expected variables to be {expected}, but got {parsed_config['variables']}"


# Test the case where the variables are not valid
def test_config_parser_variables_invalid(base_config):
    """Test the variables field with an invalid value."""
    base_config["variables"] = "temperature,wind"
    with pytest.raises(ValueError, match="Unsupported variable"):
        config_parser(base_config, section="era5-download")


def test_config_parser_generate_save_name(base_config):
    """Test that the save_name is correctly generated."""
    base_config["start_datetime"] = "2023-01-01"
    base_config["end_datetime"] = "2023-12-31"
    base_config["delta_time"] = "1d"

    parsed_config = config_parser(base_config, section="era5-download")

    expected_save_name = "2023-01-01T00_2023-12-31T00_1d.nc"
    assert (
        parsed_config["save_name"] == expected_save_name
    ), f"""Expected save_name to be {expected_save_name},
    but got {parsed_config['save_name']}"""


# ---- Test mock data ----


def test_download_era5_data_mock(base_config):
    """Test that the download_era5_data function correctly creates a mock dataset."""
    parsed_config = config_parser(base_config, section="era5-download")

    # Use the mock dataset
    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert isinstance(era5_data, xr.Dataset), "The result should be an xarray Dataset"
    assert (
        "temperature" in era5_data.variables
    ), "The dataset should contain temperature data"
    assert (
        era5_data.attrs["source"] == "Generated mock data"
    ), "The dataset should have the mock data source attribute"


def test_download_era5_data_mock_with_slicing_and_resampling(base_config):
    """
    Test the full pipeline of downloading, slicing, and
    resampling ERA5 data using a mock dataset.
    """
    base_config["start_datetime"] = "2019-01-01T06"
    base_config["end_datetime"] = "2019-01-05T18"
    base_config["delta_time"] = "6h"
    base_config["levels"] = "1000,500"
    parsed_config = config_parser(base_config, section="era5-download")

    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert era5_data.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 1, 6), "Expected start time to be 2019-01-01 06:00"
    assert era5_data.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 5, 18), "Expected end time to be 2019-01-05 18:00"
    assert list(era5_data.level.values) == [
        1000,
        500,
    ], "Expected levels to be [1000, 500]"
    assert (
        era5_data.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all(), "Expected time delta to be 6 hours"
