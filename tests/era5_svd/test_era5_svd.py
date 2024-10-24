"""
Tests for the era5_svd module.
"""

from datetime import datetime

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
        "start_datetime": "2019-01-01T06",
        "end_datetime": "2020-01-01T12",
    }


def test_config_parser(base_config):
    parsed_config = config_parser(base_config)
    assert isinstance(parsed_config, dict)
    assert parsed_config["start_datetime"] == datetime(
        2019, 1, 1, 6, 0
    ), f"""start_datetime should be {datetime(2019, 1, 1, 6, 0)}
    not {parsed_config['start_datetime']}"""
    assert parsed_config["end_datetime"] == datetime(
        2020, 1, 1, 12, 0
    ), f"""end_datetime should be {datetime(2020, 1, 1, 12, 0)}
    not {parsed_config['end_datetime']}"""


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