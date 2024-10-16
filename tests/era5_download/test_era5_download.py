"""
Tests for the era5_download module.
"""

from datetime import timedelta

from dmd_era5.era5_download.era5_download import config_parser


def test_config_parser():
    """
    Test the config_parser function.
    """

    config = {
        "source_path": "path/to/source",
        "start_date": "2020-01-01",
        "start_time": "00:00",
        "end_date": "2020-01-15",
        "end_time": "23:59",
        "delta_time": "1H",
        "variables": "temperature,humidity",
        "levels": "1000,850",
        "save_name": "era5_data.nc",
    }

    parsed_config = config_parser(config)

    assert parsed_config == {
        "source_path": "path/to/source",
        "start_datetime": "2020-01-01T00:00",
        "end_datetime": "2020-01-01T23:59",
        "delta_time": timedelta(hours=1),
        "variables": ["temperature", "humidity"],
        "levels": ["1000", "850"],
        "save_name": "2020-01-01_2020-01-15_1H.nc",
    }
