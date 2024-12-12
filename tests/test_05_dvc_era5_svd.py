import pytest
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here

from dmd_era5 import add_data_to_dvc, create_mock_era5_svd, space_coord_to_level_lat_lon
from dmd_era5.core import config_parser
from dmd_era5.era5_svd import (
    add_config_attributes,
    combine_svd_results,
    retrieve_era5_slice,
)


@pytest.mark.dependency(name="test_retrieve_era5_slice")
@pytest.mark.docker
@pytest.mark.parametrize(
    "config",
    ["era5_svd_config_a", "era5_svd_config_b", "era5_svd_config_c"],
)
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


@pytest.mark.dependency(name="test_add_era5_svd")
@pytest.mark.docker
@pytest.mark.parametrize("config", ["era5_svd_config_a", "era5_svd_config_b"])
def test_add_era5_svd_to_dvc(config, request):
    """
    Test that SVD results can be added to and tracked by
    Data Version Control (DVC).
    """
    config = request.getfixturevalue(config)
    parsed_config = config_parser(config, section="era5-svd")

    U, s, V, coords = create_mock_era5_svd(
        start_datetime=parsed_config["start_datetime"],
        end_datetime=parsed_config["end_datetime"],
        variables=parsed_config["variables"],
        levels=parsed_config["levels"],
        mean_center=parsed_config["mean_center"],
        scale=parsed_config["scale"],
        delay_embedding=parsed_config["delay_embedding"],
        n_components=parsed_config["n_components"],
    )
    svd_ds = combine_svd_results(U, s, V, coords)
    svd_ds = add_config_attributes(svd_ds, parsed_config)
    svd_ds = space_coord_to_level_lat_lon(svd_ds)
    svd_ds.to_netcdf(parsed_config["save_path"], format="NETCDF4")
    add_data_to_dvc(parsed_config["save_path"], svd_ds.attrs)
    with GitRepo(here()) as repo:
        repo.index.commit("Add SVD results to DVC")
    with DvcRepo(here()) as repo:
        repo.push()  # push to the remote DVC repository
