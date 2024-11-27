"""
Tests for the era5_svd module.
"""

from datetime import datetime, timedelta

import pytest

from dmd_era5 import config_parser
from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.era5_svd import svd_on_era5


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "variables": "temperature",
        "levels": "1000",
        "svd_type": "randomized",
        "delay_embedding": 2,
        "mean_center": False,
        "scale": False,
        "start_datetime": "2019-01-01T06",
        "end_datetime": "2020-01-01T12",
        "delta_time": "1h",
        "n_components": 10,
    }


@pytest.fixture
def mock_era5():
    return create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2020-02-01",
        variables=["temperature"],
        levels=[1000],
    )


def test_config_parser_basic(base_config):
    parsed_config = config_parser(base_config, section="era5-svd")
    assert isinstance(parsed_config, dict)
    assert parsed_config["start_datetime"] == datetime(
        2019, 1, 1, 6, 0
    ), f"""start_datetime should be {datetime(2019, 1, 1, 6, 0)}
    not {parsed_config['start_datetime']}"""
    assert parsed_config["end_datetime"] == datetime(
        2020, 1, 1, 12, 0
    ), f"""end_datetime should be {datetime(2020, 1, 1, 12, 0)}
    not {parsed_config['end_datetime']}"""
    assert parsed_config["delta_time"] == timedelta(hours=1)


@pytest.mark.parametrize(
    "field",
    [
        "source_path",
        "variables",
        "levels",
        "svd_type",
        "delay_embedding",
        "mean_center",
        "scale",
        "start_datetime",
        "end_datetime",
        "delta_time",
        "n_components",
    ],
)
def test_config_parser_missing_field(base_config, field):
    """Test the missing field error in the configuration."""
    del base_config[field]
    with pytest.raises(ValueError, match=f"Missing required field in config: {field}"):
        config_parser(base_config, section="era5-svd")


def test_config_parser_invalid_svd_type(base_config):
    """Test invalid SVD type in the configuration."""
    base_config["svd_type"] = "invalid"
    with pytest.raises(ValueError, match="Invalid SVD type in config"):
        config_parser(base_config, section="era5-svd")


@pytest.mark.parametrize("delay_embedding", [0, 1.2, "invalid"])
def test_config_parser_invalid_delay_embedding(base_config, delay_embedding):
    """Test invalid delay embedding in the configuration."""
    base_config["delay_embedding"] = delay_embedding
    with pytest.raises(ValueError, match="Invalid delay embedding in config"):
        config_parser(base_config, section="era5-svd")


@pytest.mark.parametrize("n_components", [0, 1.2, "invalid"])
def test_config_parser_invalid_n_components(base_config, n_components):
    """Test invalid number of components in the configuration."""
    base_config["n_components"] = n_components
    with pytest.raises(ValueError, match="Invalid number of components in config"):
        config_parser(base_config, section="era5-svd")


def test_svd_on_era5(base_config, mock_era5):
    parsed_config = config_parser(base_config, section="era5-svd")
    svd_on_era5(parsed_config, mock_era5)
