"""
Tests for the era5_svd module.
"""

import os
from datetime import datetime, timedelta

import pytest
import xarray as xr
from pyprojroot import here

from dmd_era5 import (
    apply_delay_embedding,
    config_parser,
    create_mock_era5,
    create_mock_era5_svd,
    flatten_era5_variables,
    standardize_data,
)
from dmd_era5.era5_svd import combine_svd_results, retrieve_era5_slice, svd_on_era5


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
        "save_data_matrix": True,
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


@pytest.fixture
def mock_era5_svd():
    return create_mock_era5_svd(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000],
        n_components=6,
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
    assert (
        parsed_config["save_name"] == "2019-01-01T06_2020-01-01T12_1h.nc"
    ), f"""save_name should be 2019-01-01T06_2020-01-01T12_1h.nc
    not {parsed_config['save_name']}"""
    assert parsed_config["save_path"] == os.path.join(
        here(), "data", "era5_svd", parsed_config["save_name"]
    ), f"""save_path should be
    {os.path.join(here(), 'data', 'era5_svd', parsed_config['save_name'])}
    not {parsed_config['save_path']}
    """
    assert parsed_config["era5_slice_path"] == os.path.join(
        here(), "data", "era5_download", parsed_config["save_name"]
    ), f"""era5_slice_path should be
    {os.path.join(here(), 'data', 'era5_download', parsed_config['save_name'])}
    not {parsed_config['era5_slice_path']}"""
    assert parsed_config["era5_svd_path"] == os.path.join(
        here(), "data", "era5_svd", parsed_config["save_name"]
    ), f"""era5_svd_path should be
    {os.path.join(here(), 'data', 'era5_svd', parsed_config['save_name'])}
    not {parsed_config['era5_svd_path']}
    """


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
        "save_data_matrix",
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
        data, _, _ = standardize_data(data, scale=parsed_config["scale"])
    data = flatten_era5_variables(data)
    data = apply_delay_embedding(data, parsed_config["delay_embedding"])
    U, s, V = svd_on_era5(data, parsed_config)
    n_samples = data.shape[0]
    n_time = data.shape[1]
    n_components = parsed_config["n_components"]
    assert U.shape == (n_samples, n_components), f"""
    Expected U to have shape ({n_samples}, {n_components}), got {U.shape}
    """
    assert s.shape == (n_components,), f"""
    Expected s to have shape ({n_components},), got {s.shape}
    """
    assert V.shape == (n_components, n_time), f"""
    Expected V to have shape ({n_components}, {n_time}), got {V.shape}
    """


def test_combine_svd_results(mock_era5_svd):
    """Test the combine_svd_results function."""
    U, s, V, coords, _ = mock_era5_svd
    da = combine_svd_results(U, s, V, coords)
    assert isinstance(da, xr.Dataset), f"Expected xr.Dataset, got {type(da)}"
    assert sorted(da.data_vars.keys()) == sorted(["U", "s", "V"]), f"""
    Expected data_vars to be ['U', 's', 'V'], got {list(da.data_vars.keys())}
    """
    assert sorted(da.U.dims) == sorted(["components", "space"]), """
    Expected U to have dims ('space', 'components')
    """
    assert sorted(da.s.dims) == [
        "components"
    ], """Expected s to have dims ('components',)"""
    assert sorted(da.V.dims) == sorted(["components", "time"]), """
    Expected V to have dims ('components', 'time')
    """
    assert sorted(da.U.coords.keys()) == sorted(
        ["space", "components", "original_variable", "delay"]
    ), """
    Expected U to have coords ['space', 'components', 'original_variable', 'delay']
    """
    assert sorted(da.s.coords.keys()) == sorted(["components"]), """
    Expected s to have coords ['components']
    """
    assert sorted(da.V.coords.keys()) == sorted(["components", "time"]), """
    Expected V to have coords ['components', 'time']
    """


def test_combine_svd_results_with_original_data(mock_era5_svd):
    """
    Test the combine_svd_results function, adding the original data array.
    """
    U, s, V, coords, X = mock_era5_svd
    da = combine_svd_results(U, s, V, coords, X=X)
    assert sorted(da.data_vars.keys()) == sorted(["U", "s", "V", "X"]), f"""
    Expected data vars to be ['U', 's', 'V', 'X'], got {list(da.data_vars.keys())}
    """
    assert da.U.shape[0] == da.X.shape[0], f"""
    Expected U and X to have the same number of rows,
    got {da.U.shape[0]} and {da.X.shape[0]}.
    """
    assert da.V.shape[1] == da.X.shape[1], f"""
    Expected V and X to have the same number of columns,
    got {da.V.shape[1]} and {da.X.shape[1]}.
    """


def test_retrieve_era5_slice_without_dvc(base_config):
    """
    Test retrieve_era5_slice returns None if the
    requested data is not found in the working directory.
    """
    parsed_config = config_parser(base_config, section="era5-svd")
    ds, _ = retrieve_era5_slice(parsed_config, use_dvc=False)
    assert ds is None, "Expected None, got a Dataset"
