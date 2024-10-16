"""
Tests for the config_reader module.
"""

import os

from pyprojroot import here

from dmd_era5.config_reader import config_reader

config_path = os.path.join(here(), "tests/config_reader/config.ini")


def test_config_reader():
    """
    Test the config_reader function.
    """

    assert os.path.exists(config_path)

    config_0 = config_reader("section-0", config_path)
    config_1 = config_reader("section-1", config_path)

    assert isinstance(config_0, dict)
    assert isinstance(config_1, dict)

    assert len(config_0) == 3
    assert len(config_1) == 4

    assert config_0["param_0"] == "value_0"
    assert config_0["param_2"] == "value_2.0,value_2.1"
    assert config_1["param_3"] == "value_3"
