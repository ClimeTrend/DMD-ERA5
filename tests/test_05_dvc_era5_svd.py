import pytest
import yaml
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

dvc_file_path = "data/era5_svd/2019-01-01T00_2019-01-01T04_1h.nc.dvc"
dvc_log_path = "data/era5_svd/2019-01-01T00_2019-01-01T04_1h.nc.yaml"
data_path = "data/era5_svd/2019-01-01T00_2019-01-01T04_1h.nc"


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


@pytest.mark.dependency(name="test_add_era5_svd")
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_svd_config_a", "era5_svd_config_b", "era5_svd_config_c"]
)
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


@pytest.mark.dependency(name="test_dvc_file_and_log", depends=["test_add_era5_svd"])
@pytest.mark.docker
def test_dvc_file_and_log():
    """
    Test that the DVC file and log are correctly created, are
    tracked by Git, and contain the expected metadata.
    """

    # check that the DVC file and log have been created and committed
    with GitRepo(here()) as repo:
        dvc_file = list(repo.iter_commits(all=True, max_count=10, paths=dvc_file_path))
        dvc_log = list(repo.iter_commits(all=True, max_count=10, paths=dvc_log_path))
    assert len(dvc_file) == 3, "There should be 3 commits for the DVC file"
    assert len(dvc_log) == 3, "There should be 3 commits for the DVC log file"

    # check that the log file contains the expected metadata
    with open(dvc_log_path) as f:
        dvc_log_content = yaml.safe_load(f)
    assert len(dvc_log_content) == 3, "The log file should contain 3 entries"
    dvc_log_keys = list(dvc_log_content.keys())
    assert dvc_log_content[dvc_log_keys[0]]["variables"] == [
        "temperature"
    ], "The first entry of the log should contain temperature data"
    assert bool(dvc_log_content[dvc_log_keys[0]]["scale"]) is False, """
    The first entry of the log should not have been scaled
    """
    assert dvc_log_content[dvc_log_keys[1]]["variables"] == [
        "u_component_of_wind"
    ], "The second entry of the log should contain u-wind data"
    assert dvc_log_content[dvc_log_keys[2]]["variables"] == [
        "temperature",
        "v_component_of_wind",
    ], "The third entry of the log should contain temperature and v-wind data"
    assert bool(dvc_log_content[dvc_log_keys[2]]["scale"]) is True, """
    The third entry of the log should have been scaled
    """
