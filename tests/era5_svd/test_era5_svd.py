"""
Tests for the era5_svd module.
"""

import pytest

from dmd_era5.era5_svd import config_parser


@pytest.fixture
def base_config():
    return {
        "file_path": "",
        "save_name": "",
        "variables": "temperature",
        "levels": "all",
        "svd_type": "randomized",
        "delay_embedding": 2,
        "standardize": False,
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
    }


def test_config_parser(base_config):
    parsed_config = config_parser(base_config)
    assert isinstance(parsed_config, dict)


@pytest.mark.parametrize(
    "field",
    [
        "file_path",
        "save_name",
        "variables",
        "levels",
        "svd_type",
        "delay_embedding",
        "standardize",
        "start_datetime",
        "end_datetime",
    ],
)
def test_config_parser_missing_field(base_config, field):
    """Test the missing field error."""
    del base_config[field]
    with pytest.raises(ValueError, match=f"Missing required field in config: {field}"):
        config_parser(base_config)
