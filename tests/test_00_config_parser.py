"""
Test the config_parser function.
"""

from datetime import timedelta

import pytest

from dmd_era5 import config_parser


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all_pressure_level_vars",
        "levels": "1000",
        "save_name": "",
    }


# Test datetime fields
@pytest.mark.parametrize(
    ("datetime_field", "invalid_datetime"),
    [
        ("start_datetime", "2019-02-31"),
        ("start_datetime", "2019-13-01"),
        ("end_datetime", "2019-01-01T25"),
    ],
)
def test_config_parser_invalid_datetime(base_config, datetime_field, invalid_datetime):
    """Test the invalid datetime error."""
    base_config[datetime_field] = invalid_datetime
    with pytest.raises(ValueError, match="Invalid datetime"):
        config_parser(base_config, section="era5-download")


# Test delta_time field
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
    parsed_config = config_parser(base_config, section="era5-download")
    assert (
        parsed_config["delta_time"] == expected
    ), f"Expected delta_time to be {expected}, but got {parsed_config['delta_time']}"


@pytest.mark.parametrize("invalid_delta", ["1v", "1", "not-a-delta"])
def test_config_parser_invalid_delta_time(base_config, invalid_delta):
    """Test the invalid delta_time error."""
    base_config["delta_time"] = invalid_delta
    with pytest.raises(ValueError, match="Error parsing delta_time"):
        config_parser(base_config, section="era5-download")
