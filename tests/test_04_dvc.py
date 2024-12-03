"""
Tests for Data Version Control (DVC) functionality.
"""

import os
import shutil

import pytest
import xarray as xr
import yaml
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here

from dmd_era5 import (
    add_data_to_dvc,
    config_parser,
    create_mock_era5,
    retrieve_data_from_dvc,
)
from dmd_era5.era5_download import (
    add_config_attributes,
)
from dmd_era5.era5_download import (
    main as era5_download_main,
)


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2019-01-01T04",
        "delta_time": "1h",
        "variables": "all",
        "levels": "1000",
    }


@pytest.fixture
def era5_data_config_a(base_config):
    config = base_config.copy()
    config["variables"] = "temperature"
    config["levels"] = "1000"
    return config


@pytest.fixture
def era5_data_config_b(base_config):
    config = base_config.copy()
    config["variables"] = "u_component_of_wind"
    config["levels"] = "900"
    return config


@pytest.fixture
def era5_data_config_c(base_config):
    config = base_config.copy()
    config["variables"] = "temperature,v_component_of_wind"
    config["levels"] = "800,700"
    return config


@pytest.fixture
def era5_data_config_d(base_config):
    config = base_config.copy()
    config["variables"] = "temperature"
    config["levels"] = "500"
    return config


dvc_file_path = "data/era5_download/2019-01-01T00_2019-01-01T04_1h.nc.dvc"
dvc_log_path = "data/era5_download/2019-01-01T00_2019-01-01T04_1h.nc.yaml"
data_path = "data/era5_download/2019-01-01T00_2019-01-01T04_1h.nc"


@pytest.mark.dependency(name="test_add_era5_to_dvc")
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_data_config_a", "era5_data_config_b", "era5_data_config_c"]
)
def test_add_era5_to_dvc(config, request):
    """
    Test that ERA5 slices can be added to and tracked by
    Data Version Control (DVC).
    """
    config = request.getfixturevalue(config)
    parsed_config = config_parser(config, section="era5-download")

    era5_ds = create_mock_era5(
        start_datetime=parsed_config["start_datetime"],
        end_datetime=parsed_config["end_datetime"],
        variables=parsed_config["variables"],
        levels=parsed_config["levels"],
    )
    era5_ds = add_config_attributes(era5_ds, parsed_config)
    era5_ds.to_netcdf(parsed_config["save_path"], format="NETCDF4")
    add_data_to_dvc(parsed_config["save_path"], era5_ds.attrs)
    with GitRepo(here()) as repo:
        repo.index.commit("Add ERA5 data to DVC")
    with DvcRepo(here()) as repo:
        repo.push()  # push to the remote DVC repository


@pytest.mark.dependency(name="test_dvc_file_and_log", depends=["test_add_era5_to_dvc"])
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
    assert dvc_log_content[dvc_log_keys[1]]["variables"] == [
        "u_component_of_wind"
    ], "The second entry of the log should contain u-wind data"
    assert dvc_log_content[dvc_log_keys[2]]["variables"] == [
        "temperature",
        "v_component_of_wind",
    ], "The third entry of the log should contain temperature and v-wind data"


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
    data_vars = list(data.data_vars)
    assert sorted(data_vars) == ["temperature", "v_component_of_wind"], """
    The data should contain temperature and v-wind data."""
    data.close()

    # checkout the first commit of the DVC file
    # to test that the data version has temperature data
    with GitRepo(here()) as repo:
        repo.git.checkout("HEAD~2", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    data = xr.open_dataset(data_path)
    data_vars = next(iter(data.data_vars))
    assert data_vars == "temperature", "The data should contain temperature data"
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
    name="test_dvc_retrieve_era5_data", depends=["test_dvc_data_versions"]
)
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_data_config_a", "era5_data_config_b", "era5_data_config_c"]
)
def test_dvc_retrieve_era5_data(config, request):
    """
    Test that the correct ERA5 slice can be retrieved from DVC
    using the retrieve_data_from_dvc function and the parsed
    configuration.
    """
    config = request.getfixturevalue(config)
    parsed_config = config_parser(config, section="era5-download")
    retrieve_data_from_dvc(parsed_config)
    data = xr.open_dataset(parsed_config["save_path"])
    data_vars = list(data.data_vars)
    levels = data.level.values.tolist()
    data.close()

    # Git restore the DVC file, which will have been checked out
    # by retrieve_data_from_dvc
    with GitRepo(here()) as repo:
        repo.git.restore("--staged", dvc_file_path)
        repo.git.restore(dvc_file_path)
        diff = repo.git.diff("HEAD", dvc_file_path)
    with DvcRepo(here()) as repo:
        repo.checkout()
    assert diff == "", "The DVC file should have been Git restored"

    for var in parsed_config["variables"]:
        assert var in data_vars, f"The data should contain {var} data"
    for level in parsed_config["levels"]:
        assert level in levels, f"The data should contain level {level} data"


@pytest.mark.dependency(
    name="test_main_era5_download", depends=["test_dvc_retrieve_era5_data"]
)
@pytest.mark.docker
@pytest.mark.parametrize(
    "config", ["era5_data_config_a", "era5_data_config_b", "era5_data_config_d"]
)
def test_main_era5_download(config, request):
    """
    Test the main function for downloading ERA5 data,
    adding data to DVC, and retrieving data from DVC.

    The expectation is that provided configurations A and B,
    the data should be retrieved from DVC, and that the data
    should contain the expected variables and levels. For
    configuration D, the data should be added to DVC, and
    the DVC log should be updated with the new entry.
    """
    config_dict = request.getfixturevalue(config)
    parsed_config = config_parser(config_dict, section="era5-download")
    dvc_file_path = parsed_config["save_path"] + ".dvc"
    dvc_log_path = parsed_config["save_path"] + ".yaml"

    if config != "era5_data_config_d":
        added_to_dvc, retrieved_from_dvc = era5_download_main(
            config_dict, use_mock_data=True, use_dvc=True
        )
        assert (
            not added_to_dvc
        ), "The data should not have been added to DVC as it already exists."
        assert retrieved_from_dvc, "The data should have been retrieved from DVC."

        data = xr.open_dataset(parsed_config["save_path"])
        data_vars = list(data.data_vars)
        levels = data.level.values.tolist()
        data.close()

        if config == "era5_data_config_a":
            assert (
                data_vars[0] == "temperature"
            ), "The data should contain temperature data"
            assert (
                levels[0] == 1000
            ), "The data should contain data at pressure level 1000"
        elif config == "era5_data_config_b":
            assert (
                data_vars[0] == "u_component_of_wind"
            ), "The data should contain u-wind data"
            assert (
                levels[0] == 900
            ), "The data should contain data at pressure level 900"

        # Git restore the DVC file, which will have been checked out
        # by when retrieving data from DVC. The DVC log should not be changed.
        with GitRepo(here()) as repo:
            repo.git.restore("--staged", dvc_file_path)
            repo.git.restore(dvc_file_path)
            diff_dvc_file = repo.git.diff("HEAD", dvc_file_path)
            diff_dvc_log = repo.git.diff("HEAD", dvc_log_path)
        with DvcRepo(here()) as repo:
            repo.checkout()
        assert diff_dvc_file == "", "The DVC file should have been Git restored"
        assert diff_dvc_log == "", "The DVC log file should not have been changed"

    else:
        added_to_dvc, retrieved_from_dvc = era5_download_main(
            config_dict, use_mock_data=True, use_dvc=True
        )
        assert added_to_dvc, "The data should have been added to DVC."
        assert (
            not retrieved_from_dvc
        ), "The data should not have been retrieved from DVC."

        # Make sure the a new entry has been added to the DVC log
        # but that the DVC file has not been changed because we haven't
        # saved the mock data to disk
        with GitRepo(here()) as repo:
            diff_dvc_log = repo.git.diff("HEAD", dvc_log_path)
            diff_dvc_file = repo.git.diff("HEAD", dvc_file_path)
        assert diff_dvc_log != "", "The DVC log file should have been changed"
        assert diff_dvc_file == "", "The DVC file should not have been changed"

        # Restore the DVC log
        with GitRepo(here()) as repo:
            repo.git.restore("--staged", dvc_log_path)
            repo.git.restore(dvc_log_path)
            diff_dvc_log = repo.git.diff("HEAD", dvc_log_path)
        assert diff_dvc_log == "", "The DVC log file should have been Git restored"


@pytest.mark.dependency(
    name="test_dvc_retrieval_from_remote",
    depends=["test_add_era5_to_dvc", "test_main_era5_download"],
)
@pytest.mark.docker
def test_dvc_retrieval_from_remote(era5_data_config_b):
    """
    Test that data can be retrieved from the remote DVC repository.

    Since the data is pushed to the remote DVC repository, even if it
    is deleted locally, it should be possible to retrieve it.
    """

    # delete the local data file and the DVC cache
    os.remove(data_path)
    shutil.rmtree(".dvc/cache")
    assert not os.path.exists(data_path), "The data file should have been deleted"
    assert not os.path.exists(".dvc/cache"), "The DVC cache should have been deleted"

    added_to_dvc, retrieved_from_dvc = era5_download_main(
        era5_data_config_b, use_mock_data=True, use_dvc=True
    )
    assert not added_to_dvc, "The data should not have been added to DVC"
    assert retrieved_from_dvc, "The data should have been retrieved from DVC"

    data = xr.open_dataset(data_path)
    data_vars = list(data.data_vars)
    levels = data.level.values.tolist()
    data.close()

    assert sorted(data_vars) == ["u_component_of_wind"]
    assert sorted(levels) == [900]
