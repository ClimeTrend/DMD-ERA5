"""
Tests for the era5_svd module.
"""

from dmd_era5.era5_svd import config_parser


def test_config_parser():
    config = config_parser()
    assert isinstance(config, dict)
