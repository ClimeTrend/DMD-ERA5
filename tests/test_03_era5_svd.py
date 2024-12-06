"""
Tests for the era5_svd module.
"""

from datetime import datetime, timedelta

import pytest

from dmd_era5 import config_parser
from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.era5_svd import svd_on_era5
from dmd_era5.slice_tools import (
    apply_delay_embedding,
    flatten_era5_variables,
    standardize_data,
)


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


@pytest.fixture
def mock_era5_small():
    return create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
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


@pytest.mark.parametrize("svd_type", ["standard", "randomized"])
def test_svd_on_era5(base_config, mock_era5_small, svd_type):
    """Test the svd_on_era5 function."""
    config = base_config.copy()
    data = mock_era5_small
    config["svd_type"] = svd_type
    parsed_config = config_parser(base_config, section="era5-svd")
    if parsed_config["mean_center"]:
        data = standardize_data(data, scale=parsed_config["scale"])
    data = flatten_era5_variables(data)
    data = apply_delay_embedding(data, parsed_config["delay_embedding"])
    U, s, V = svd_on_era5(data, parsed_config)
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_components = parsed_config["n_components"]
    assert U.shape == (n_samples, n_components)
    assert s.shape == (n_components,)
    assert V.shape == (n_components, n_features)
