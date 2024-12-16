import numpy as np
import pytest
import xarray as xr
import yaml
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here

from dmd_era5 import (
    add_data_to_dvc,
    create_mock_era5_svd,
    space_coord_to_level_lat_lon,
)
from dmd_era5.core import config_parser
from dmd_era5.era5_svd import (
    add_config_attributes,
    combine_svd_results,
    retrieve_era5_slice,
    retrieve_svd_results,
)
from dmd_era5.era5_svd import (
    main as era5_svd_main,
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


@pytest.mark.dependency(name="test_dvc_md5_hashes", depends=["test_dvc_file_and_log"])
@pytest.mark.docker
def test_dvc_md5_hashes():
    """
    Test that the DVC md5 hashes are correctly set in the DVC log
    """

    # the last log entry should contain the same md5 hash as
    # the last commit of the DVC file
    with open(dvc_file_path) as f:
        dvc_file_content = yaml.safe_load(f)
    md5_hash = dvc_file_content["outs"][0]["md5"]
    with open(dvc_log_path) as f:
        dvc_log_content = yaml.safe_load(f)
    dvc_log_keys = list(dvc_log_content.keys())
    assert (
        md5_hash == dvc_log_keys[-1]
    ), "The md5 hash in the last commit of the DVC file should match the last log entry"

    # the first log entry should contain the same md5 hash as
    # the first commit of the DVC file
    # checkout the first commit of the DVC file
    with GitRepo(here()) as repo:
        repo.git.checkout("HEAD~2", dvc_file_path)
    with open(dvc_file_path) as f:
        dvc_file_content = yaml.safe_load(f)
    md5_hash = dvc_file_content["outs"][0]["md5"]
    assert md5_hash == dvc_log_keys[0], """
    The md5 hash in the first commit of the DVC file
    should match the first log entry
    """

    # restore the DVC file
    with GitRepo(here()) as repo:
        repo.git.restore("--staged", dvc_file_path)
        repo.git.restore(dvc_file_path)
        diff = repo.git.diff("HEAD", dvc_file_path)  # check that the file is restored

    assert diff == "", "The DVC file should have been Git restored"


@pytest.mark.dependency(name="test_dvc_data_versions", depends=["test_dvc_md5_hashes"])
@pytest.mark.docker
def test_dvc_data_versions():
    """
    Test that the data versions are correctly tracked by DVC
    """

    # test that the current version has temperature and v-wind data
    data = xr.open_dataset(data_path)
    data_vars = sorted(np.unique(data.coords["original_variable"].values).tolist())
    data_vars_attrs = sorted(data.attrs["variables"])
    assert data_vars == data_vars_attrs, """
    The variables in the coordinates should match the variables in the attributes
    """
    assert data_vars == ["temperature", "v_component_of_wind"], """
    The data should contain temperature and v-wind data."""
    data.close()

    # checkout the first commit of the DVC file
    # to test that the data version has temperature data
    with GitRepo(here()) as repo:
        repo.git.checkout("HEAD~2", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    data = xr.open_dataset(data_path)
    data_vars = np.unique(data.coords["original_variable"].values).tolist()
    data_vars_attrs = [data.attrs["variables"]]
    assert data_vars == data_vars_attrs, """
    The variables in the coordinates should match the variables in the attributes
    """
    assert data_vars == ["temperature"], "The data should contain temperature data"
    data.close()

    # restore the DVC file
    with GitRepo(here()) as repo:
        repo.git.restore("--staged", dvc_file_path)
        repo.git.restore(dvc_file_path)
        diff = repo.git.diff("HEAD", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    assert diff == "", "The DVC file should have been Git restored"


@pytest.mark.dependency(
    name="test_retrieve_svd_results", depends=["test_dvc_data_versions"]
)
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_svd_config_c", "era5_svd_config_a", "era5_svd_config_b"]
)
def test_retrieve_svd_results(config, request):
    """Test the retrieve_svd_results function using DVC."""
    config_dict = request.getfixturevalue(config)
    parsed_config = config_parser(config_dict, "era5-svd")
    svd_ds, retrieved_from_dvc = retrieve_svd_results(parsed_config, use_dvc=True)
    if config == "era5_svd_config_c":
        # should not be retrieved from DVC because this version should be
        # in the working directory
        assert retrieved_from_dvc is False
    else:
        assert retrieved_from_dvc is True
    data_vars = np.unique(svd_ds.coords["original_variable"].values).tolist()
    levels = np.unique(svd_ds.coords["level"].values).tolist()
    mean_center = bool(svd_ds.attrs["mean_center"])
    scale = bool(svd_ds.attrs["scale"])
    assert sorted(data_vars) == sorted(parsed_config["variables"]), """
    The data should contain the expected variables
    """
    assert sorted(levels) == sorted(parsed_config["levels"]), """
    The data should contain the expected pressure levels
    """
    assert mean_center == parsed_config["mean_center"], """
    The data should have the expected mean centering
    """
    assert scale == parsed_config["scale"], """
    The data should have the expected scaling
    """

    # Git restore the DVC file, which will have been checked out
    # by retrieve_data_from_dvc
    with GitRepo(here()) as repo:
        repo.git.restore("--staged", dvc_file_path)
        repo.git.restore(dvc_file_path)
        diff = repo.git.diff("HEAD", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    assert diff == "", "The DVC file should have been Git restored"


@pytest.mark.dependency(
    name="test_era5_svd_main_retrieved_results", depends=["test_retrieve_svd_results"]
)
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_svd_config_a", "era5_svd_config_b", "era5_svd_config_c"]
)
def test_era5_svd_main_retrieved_results(config, request):
    """
    Test the main function of era5_svd with results
    retrieved from DVC or the working directory.
    """
    config_dict = request.getfixturevalue(config)
    _, added_to_dvc, retrieved_from_dvc = era5_svd_main(config_dict, use_dvc=True)
    assert added_to_dvc is False, "The results should not have been added to DVC"
    if config == "era5_svd_config_c":
        # should not be retrieved from DVC because this version should be
        # in the working directory
        assert retrieved_from_dvc is False, """
        The results should not have been retrieved from DVC
        """
    else:
        assert retrieved_from_dvc is True, """
        The results should have been retrieved from DVC
        """

    # Git restore the DVC file, which will have been checked out
    # by retrieve_data_from_dvc
    with GitRepo(here()) as repo:
        repo.git.restore("--staged", dvc_file_path)
        repo.git.restore(dvc_file_path)
        diff = repo.git.diff("HEAD", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    assert diff == "", "The DVC file should have been Git restored"
