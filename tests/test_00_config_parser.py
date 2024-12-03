"""
Test the config_parser function.
"""

from datetime import timedelta

import pytest

from dmd_era5 import config_parser


@pytest.fixture
def base_config_era5_download():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all_pressure_level_vars",
        "levels": "1000",
    }


@pytest.fixture
def base_config_era5_svd():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all_pressure_level_vars",
        "levels": "1000",
        "svd_type": "randomized",
        "delay_embedding": 2,
        "mean_center": True,
        "scale": True,
        "n_components": 10,
    }


@pytest.mark.parametrize(
    ("datetime_field", "invalid_datetime"),
    [
        ("start_datetime", "2019-02-31"),
        ("start_datetime", "2019-13-01"),
        ("end_datetime", "2019-01-01T25"),
    ],
)
@pytest.mark.parametrize("section", ["era5-download", "era5-svd"])
def test_config_parser_invalid_datetime(
    base_config_era5_download,
    base_config_era5_svd,
    datetime_field,
    invalid_datetime,
    section,
):
    """Test the invalid datetime error."""
    if section == "era5-download":
        config = base_config_era5_download
    elif section == "era5-svd":
        config = base_config_era5_svd
    config[datetime_field] = invalid_datetime
    with pytest.raises(ValueError, match="Invalid datetime"):
        config_parser(config, section=section)


@pytest.mark.parametrize(
    ("delta_time", "expected"),
    [
        ("1h", timedelta(hours=1)),
        ("24h", timedelta(hours=24)),
        ("1d", timedelta(days=1)),
        ("7d", timedelta(days=7)),
    ],
)
@pytest.mark.parametrize("section", ["era5-download", "era5-svd"])
def test_config_parser_delta_time(
    base_config_era5_download,
    base_config_era5_svd,
    delta_time,
    expected,
    section,
):
    """Test the delta_time field."""
    if section == "era5-download":
        config = base_config_era5_download
    elif section == "era5-svd":
        config = base_config_era5_svd
    config["delta_time"] = delta_time
    parsed_config = config_parser(config, section=section)
    assert (
        parsed_config["delta_time"] == expected
    ), f"Expected delta_time to be {expected}, but got {parsed_config['delta_time']}"


@pytest.mark.parametrize("invalid_delta", ["1v", "1", "not-a-delta"])
@pytest.mark.parametrize("section", ["era5-download", "era5-svd"])
def test_config_parser_invalid_delta_time(
    base_config_era5_download,
    base_config_era5_svd,
    invalid_delta,
    section,
):
    """Test the invalid delta_time error."""
    if section == "era5-download":
        config = base_config_era5_download
    elif section == "era5-svd":
        config = base_config_era5_svd
    config["delta_time"] = invalid_delta
    with pytest.raises(ValueError, match="Error parsing delta_time"):
        config_parser(config, section=section)
