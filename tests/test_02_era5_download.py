"""
Tests for the era5_download module.
"""

from datetime import datetime, timedelta

import pytest
import xarray as xr
import yaml
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here

from dmd_era5.dvc_tools import add_data_to_dvc, retrieve_data_from_dvc
from dmd_era5.era5_download import (
    add_config_attributes,
    config_parser,
    download_era5_data,
)
from dmd_era5.era5_download import (
    main as era5_download_main,
)
from dmd_era5.utils import create_mock_era5


@pytest.fixture
def base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2020-01-01T00",
        "delta_time": "1y",
        "variables": "all",
        "levels": "1000",
    }


def test_config_parser_basic(base_config):
    parsed_config = config_parser(base_config)

    assert (
        parsed_config["source_path"] == base_config["source_path"]
    ), f"""source_path should be {base_config['source_path']}
    not {parsed_config['source_path']}"""
    assert parsed_config["start_datetime"] == datetime(
        2019, 1, 1
    ), f"""start_datetime should be {datetime(2019, 1, 1, 0, 0)}
    not {parsed_config['start_datetime']}"""
    assert parsed_config["end_datetime"] == datetime(
        2020, 1, 1
    ), f"""end_datetime should be {datetime(2019, 1, 2, 0, 0)}
    not {parsed_config['end_datetime']}"""
    assert parsed_config["delta_time"] == timedelta(
        days=365
    ), f"delta_time should be {timedelta(hours=1)} not {parsed_config['delta_time']}"
    assert parsed_config["variables"] == [
        "all"
    ], f"variables should be ['all'] not {parsed_config['variables']}"
    assert parsed_config["levels"] == [
        1000
    ], f"levels should be [1000] not {parsed_config['levels']}"
    assert (
        parsed_config["save_name"] == "2019-01-01T00_2020-01-01T00_1y.nc"
    ), f"""save_name should be 2019-01-01T00_2020-01-01T00_1y.nc
    not {parsed_config['save_name']}"""


# ----- Test cases -----


# --- Missing field
@pytest.mark.parametrize(
    "field",
    [
        "source_path",
        "start_datetime",
        "end_datetime",
        "delta_time",
        "variables",
        "levels",
    ],
)
def test_config_parser_missing_field(base_config, field):
    """Test the missing field error."""
    del base_config[field]
    with pytest.raises(ValueError, match=f"Missing required field in config: {field}"):
        config_parser(base_config)


# --- Invalid datetime
@pytest.mark.parametrize(
    ("datetime_field", "invalid_datetime"),
    [
        ("start_datetime", "2019-02-31"),
        ("start_datetime", "2019-13-01"),
        ("start_datetime", "2019-01-01T25"),
    ],
)
def test_config_parser_invalid_datetime(base_config, datetime_field, invalid_datetime):
    """Test the invalid datetime error."""
    base_config[datetime_field] = invalid_datetime
    with pytest.raises(ValueError, match="Invalid start"):
        config_parser(base_config)


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ("1000,850,500", [1000, 850, 500]),
        ("1000", [1000]),
        (" 1000 , 850 ", [1000, 850]),
    ],
)
def test_config_parser_levels(base_config, levels, expected):
    """Test the levels field."""
    base_config["levels"] = levels
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["levels"] == expected
    ), f"Expected levels to be {expected}, but got {parsed_config['levels']}"


@pytest.mark.parametrize(
    ("delta_time", "expected"),
    [
        ("1h", timedelta(hours=1)),
        ("24h", timedelta(hours=24)),
        ("1d", timedelta(days=1)),
        ("7d", timedelta(days=7)),
    ],
)
def test_config_parser_delta_time(base_config, delta_time, expected):
    """Test the delta_time field."""
    base_config["delta_time"] = delta_time
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["delta_time"] == expected
    ), f"Expected delta_time to be {expected}, but got {parsed_config['delta_time']}"


@pytest.mark.parametrize("invalid_delta", ["1v", "1", "not-a-delta"])
def test_config_parser_invalid_delta_time(base_config, invalid_delta):
    """Test the invalid delta_time error."""
    base_config["delta_time"] = invalid_delta
    with pytest.raises(ValueError, match="Error parsing delta_time"):
        config_parser(base_config)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ("temperature,humidity,pressure", ["temperature", "humidity", "pressure"]),
        ("all", ["all"]),
        (" temperature , humidity ", ["temperature", "humidity"]),
    ],
)
def test_config_parser_variables(base_config, variables, expected):
    """Test the variables field."""
    base_config["variables"] = variables
    parsed_config = config_parser(base_config)
    assert (
        parsed_config["variables"] == expected
    ), f"Expected variables to be {expected}, but got {parsed_config['variables']}"


def test_config_parser_generate_save_name(base_config):
    """Test that the save_name is correctly generated."""
    base_config["start_datetime"] = "2023-01-01"
    base_config["end_datetime"] = "2023-12-31"
    base_config["delta_time"] = "1d"

    parsed_config = config_parser(base_config)

    expected_save_name = "2023-01-01T00_2023-12-31T00_1d.nc"
    assert (
        parsed_config["save_name"] == expected_save_name
    ), f"""Expected save_name to be {expected_save_name},
    but got {parsed_config['save_name']}"""


# ---- Test mock data ----


def test_download_era5_data_mock(base_config):
    """Test that the download_era5_data function correctly creates a mock dataset."""
    parsed_config = config_parser(base_config)

    # Use the mock dataset
    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert isinstance(era5_data, xr.Dataset), "The result should be an xarray Dataset"
    assert (
        "temperature" in era5_data.variables
    ), "The dataset should contain temperature data"
    assert (
        era5_data.attrs["source"] == "Generated mock data"
    ), "The dataset should have the mock data source attribute"


def test_download_era5_data_mock_with_slicing_and_thinning(base_config):
    """
    Test the full pipeline of downloading, slicing, and
    thinning ERA5 data using a mock dataset.
    """
    base_config["start_datetime"] = "2019-01-01T06"
    base_config["end_datetime"] = "2019-01-05T18"
    base_config["delta_time"] = "6h"
    base_config["levels"] = "1000,500"
    parsed_config = config_parser(base_config)

    era5_data = download_era5_data(parsed_config, use_mock_data=True)

    assert era5_data.time.min().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 1, 6)
    assert era5_data.time.max().values.astype("datetime64[us]").astype(
        datetime
    ) == datetime(2019, 1, 5, 18)
    assert list(era5_data.level.values) == [1000, 500]
    assert (
        era5_data.time.diff("time").astype("timedelta64[ns]").astype(int)
        == 6 * 3600 * 1e9
    ).all()


@pytest.fixture
def era5_data_config_a(base_config):
    config = base_config.copy()
    config["start_datetime"] = "2019-01-01T00"
    config["end_datetime"] = "2019-01-01T04"
    config["delta_time"] = "1h"
    config["variables"] = "temperature"
    config["levels"] = "1000"
    return config


@pytest.fixture
def era5_data_config_b(base_config):
    config = base_config.copy()
    config["start_datetime"] = "2019-01-01T00"
    config["end_datetime"] = "2019-01-01T04"
    config["delta_time"] = "1h"
    config["variables"] = "u_component_of_wind"
    config["levels"] = "900"
    return config


@pytest.fixture
def era5_data_config_c(base_config):
    config = base_config.copy()
    config["start_datetime"] = "2019-01-01T00"
    config["end_datetime"] = "2019-01-01T04"
    config["delta_time"] = "1h"
    config["variables"] = "temperature,v_component_of_wind"
    config["levels"] = "800,700"
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
    parsed_config = config_parser(config)

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
    parsed_config = config_parser(config)
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


@pytest.fixture
def era5_data_config_d(base_config):
    config = base_config.copy()
    config["start_datetime"] = "2019-01-01T00"
    config["end_datetime"] = "2019-01-01T04"
    config["delta_time"] = "1h"
    config["variables"] = "temperature"
    config["levels"] = "500"
    return config


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
    parsed_config = config_parser(config_dict)
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
