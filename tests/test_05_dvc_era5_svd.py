import pytest

from dmd_era5 import config_parser
from dmd_era5.era5_svd import retrieve_era5_slice


@pytest.mark.dependency(name="test_retrieve_era5_slice")
@pytest.mark.docker
@pytest.mark.parametrize("config", ["era5_svd_config_a", "era5_svd_config_b"])
def test_retrieve_era5_slice(config, request):
    """Test the retrieve_era5_slice function using DVC."""
    config_dict = request.getfixturevalue(config)
    parsed_config = config_parser(config_dict, "era5-svd")
    era5_ds, retrieved_from_dvc = retrieve_era5_slice(parsed_config, use_dvc=True)
    assert retrieved_from_dvc is True
    data_vars = list(era5_ds.data_vars)
    levels = era5_ds.level.values.tolist()
    if config == "era5_svd_config_a":
        assert "temperature" in data_vars
        assert "u_component_of_wind" not in data_vars
        assert 1000 in levels
        assert 850 not in levels
    if config == "era5_svd_config_b":
        assert "u_component_of_wind" in data_vars
        assert "temperature" not in data_vars
        assert 850 in levels
        assert 1000 not in levels
