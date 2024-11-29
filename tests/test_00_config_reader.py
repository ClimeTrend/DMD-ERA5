"""
Tests for the config_reader module.
"""

import os

import pytest
from pyprojroot import here

from dmd_era5.config_reader import config_reader


# ---- Fixtures ----
@pytest.fixture
def test_config_path():
    """Fixture for the test config path."""
    return os.path.join(here(), "tests/config.ini")


@pytest.fixture
def actual_config_path():
    """Fixture for the current config path in the project."""
    return os.path.join(here(), "config.ini")


@pytest.fixture
def test_config_reader(test_config_path):
    """Fixture for reading the test configuration."""

    def test_config_reader(section):
        return config_reader(section, test_config_path)

    return test_config_reader


@pytest.fixture
def actual_config_reader(actual_config_path):
    """Fixture for reading the actual configuration."""

    def actual_config_reader(section):
        return config_reader(section, actual_config_path)

    return actual_config_reader


# ---- Tests ----


def test_config_files_exists(test_config_path, actual_config_path):
    """Test that the test and actual config files exist."""
    assert os.path.exists(test_config_path), "Test config file does not exist."
    assert os.path.exists(actual_config_path), "Config file does not exist in project."


def test_config_reader_returns_dict(test_config_reader):
    """Test that the config_reader function returns a dictionary."""
    config = test_config_reader("test-section-0")
    assert isinstance(config, dict), "Expected config_reader to return a dictionary."


def test_config_reader_sections(test_config_reader):
    """
    Test that the config_reader function returns the correct
    number of parameters for a given section.
    """
    config_section_0 = test_config_reader("test-section-0")
    assert (
        len(config_section_0) == 3
    ), "Expected 3 parameters in test-section-0 from test config file."
    assert (
        config_section_0["param_0"] == "value_0"
    ), "Expected param_0 to be 'value_0' in test-section-0 from test config file."
    assert (
        config_section_0["param_1"] == "value_1"
    ), "Expected param_1 to be 'value_1' in test-section-0 from test config file."
    assert (
        config_section_0["param_2"] == "value_2.0,value_2.1"
    ), """Expected param_2 to be 'value_2.0,value_2.1'
    in test-section-0 from test config file."""

    config_section_1 = test_config_reader("test-section-1")
    assert (
        len(config_section_1) == 4
    ), "Expected 4 parameters in test-section-1 from test config file."
    assert (
        config_section_1["param_0"] == "value_0"
    ), "Expected param_0 to be 'value_0' in test-section-1 from test config file."
    assert (
        config_section_1["param_1"] == "value_1"
    ), "Expected param_1 to be 'value_1' in test-section-1 from test config file."
    assert (
        config_section_1["param_2"] == "value_2"
    ), "Expected param_2 to be 'value_2' in test-section-1 from test config file."
    assert (
        config_section_1["param_3"] == "value_3"
    ), "Expected param_3 to be 'value_3' in test-section-1 from test config file."


def test_config_nonexistent_section(test_config_reader):
    """Test that the config_reader function returns None for a nonexistent section."""
    with pytest.raises(Exception, match="Section nonexistent-section not found"):
        test_config_reader("nonexistent-section")


# ---- Testing Current Config ----


def test_actual_config_era5_download_section(actual_config_reader):
    """Test the contents of era5-download section in the actual configuration."""
    config = actual_config_reader("era5-download")
    assert "source_path" in config, "era5-download section should have a source_path"
    assert (
        "start_datetime" in config
    ), "era5-download section should have a start_datetime"
    assert "end_datetime" in config, "era5-download section should have an end_datetime"
    assert "delta_time" in config, "era5-download section should have a delta_time"
    assert "variables" in config, "era5-download section should have variables"
    assert "levels" in config, "era5-download section should have levels"


def test_actual_config_type(actual_config_reader):
    """Test type conversion of configuration values in the actual configuration."""
    config = actual_config_reader("era5-download")

    for key, value in config.items():
        assert isinstance(value, str), f"Expected {key} to be a string."
