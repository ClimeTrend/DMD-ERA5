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
    }


def test_config_parser():
    config = config_parser()
    assert isinstance(config, dict)
