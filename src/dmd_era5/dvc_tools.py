import os
import subprocess
from datetime import datetime

import yaml
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here


def add_config_to_dvc_log(
    dvc_file_path: str, data_path, data_attrs: dict, git_add=False
) -> None:
    """
    Add the attributes of a dataset as metadata to a custom log file.
    Each entry in the log file stores metadata under a unique DVC md5 hash.
    The log file is a YAML file with the same name as the data file.

    Args:
        dvc_file_path (str): The path to the DVC file.
        data_path (str): The path to the data file.
        data_attrs (dict): The attributes of the data file.
        git_add (bool): Whether to stage the log file for commit.
    """

    # get the md5 hash of the dvc file
    with open(dvc_file_path) as f:
        dvc_file_content = yaml.safe_load(f)
    md5_hash = dvc_file_content["outs"][0]["md5"]

    log_file = data_path + ".yaml"

    # Create the log file if it does not exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    # Add the metadata to the log file
    with open(log_file, "a") as f:
        f.write(f"{md5_hash}:\n")
        for key, value in data_attrs.items():
            f.write(f"  {key}: {value}\n")

    # Stage the log file for commit
    if git_add:
        with GitRepo(here()) as repo:
            repo.index.add([log_file])


def add_data_to_dvc(data_path: str, data_attrs: dict) -> None:
    """
    Add data to Data Version Control (DVC) and log the metadata
    to a custom log file.

    Args:
        data_path (str): The path to the data file.
        data_attrs (dict): The attributes of the data file.
    """

    with DvcRepo(here()) as repo:
        repo.add(data_path)
    dvc_file_path = os.path.join(data_path + ".dvc")
    add_config_to_dvc_log(dvc_file_path, data_path, data_attrs, git_add=True)


def find_first_commit_with_md5_hash(md5_hash: str, dvc_file_path: str) -> str | None:
    """
    Find the first commit that contains a given DVC md5 hash.

    Args:
        md5_hash (str): The DVC md5 hash.
        dvc_file_path (str): The path to the DVC file.

    Returns:
        str: The commit hash.
    """

    command = [
        "git",
        "log",
        "-S",
        md5_hash,
        "--reverse",
        "--oneline",
        "--",
        dvc_file_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    output_lines = result.stdout.strip().splitlines()
    if output_lines:
        return output_lines[0].split()[0]
    return None


def fetch_data_from_default_remote(repo: DvcRepo, targets: list) -> tuple:
    """
    Fetch data from the default remote DVC repository.

    Args:
        repo (DvcRepo): The DVC repository.
        targets (list): The DVC targets to fetch.

    Returns:
        bool: Whether the remote exists.
        bool: Whether the data was fetched successfully.
    """

    remotes = repo.config["remote"]
    remote_exists = False
    data_fetched = False
    if remotes:
        remote_exists = True
        num_files = repo.fetch(targets=targets)
        if num_files > 0:
            data_fetched = True
    return remote_exists, data_fetched


def retrieve_data_from_dvc(
    parsed_config: dict,
    data_type: str = "era5_slice",
) -> None:
    """
    Given a parsed configuration, retrieve the correct version
    of the data from DVC, if it exists.
    If the dvc file or log file does not exist, raises FileNotFoundError.
    If a matching version of the data does not exist in the log, raises ValueError.
    If a matching version of the data exists in the log but cannot be retrieved
    from DVC,raises ValueError.

    Args:
        parsed_config (dict): The parsed configuration.
        data_type (str): The type of data to retrieve.
            Currently only "era5_slice" is supported.
    """
    log_file_path = parsed_config["save_path"] + ".yaml"
    dvc_file_path = parsed_config["save_path"] + ".dvc"

    if not os.path.exists(log_file_path) or not os.path.exists(dvc_file_path):
        msg = "DVC file or log file does not exist."
        raise FileNotFoundError(msg)

    with open(log_file_path) as f:
        log_file_content = yaml.safe_load(f)

    # Find the most recent version of the data that matches the requested
    # variables, levels, and source path in the configuration, by looping
    # through the log file content.
    md5_hash_keep = None  # The md5 hash of the version to keep
    date_downloaded_keep = datetime(1970, 1, 1)

    if data_type == "era5_slice":
        for md5_hash, metadata in log_file_content.items():
            if (
                sorted(parsed_config["variables"])
                == sorted(set(metadata["variables"]) & set(parsed_config["variables"]))
                and sorted(parsed_config["levels"])
                == sorted(set(metadata["levels"]) & set(parsed_config["levels"]))
                and parsed_config["source_path"] == metadata["source_path"]
            ):
                date_downloaded = metadata["date_downloaded"]
                if date_downloaded > date_downloaded_keep:
                    md5_hash_keep = md5_hash
                    date_downloaded_keep = date_downloaded
    else:
        msg = "Data type not supported."
        raise ValueError(msg)

    if md5_hash_keep:
        commit_hash = find_first_commit_with_md5_hash(md5_hash_keep, dvc_file_path)
        if commit_hash is None:
            msg = """
            Found a matching version of the data in the log file,
            but could not retrieve it from DVC.
            """
            raise ValueError(msg)
        with GitRepo(here()) as repo:
            repo.git.checkout(commit_hash, dvc_file_path)
        with DvcRepo(here()) as repo:
            if os.path.exists(os.path.join(here(), ".dvc/cache")):
                checked_out_files = repo.checkout(targets=[dvc_file_path])
                print("Checked out files:", checked_out_files)
            else:
                print(
                    "No DVC cache found. Attempting to fetch data from default remote."
                )
                remote_exists, data_fetched = fetch_data_from_default_remote(
                    repo, targets=[dvc_file_path]
                )
                if data_fetched:
                    print("Data successfully fetched.")
                    checked_out_files = repo.checkout(targets=[dvc_file_path])
                    print("Checked out files:", checked_out_files)
                if not remote_exists or not data_fetched:
                    print("Could not fetch data from default remote DVC repository.")
                    msg = """
                    Found a matching version of the data in the log file,
                    but could not retrieve it from DVC.
                    """
                    raise ValueError(msg)
    else:
        msg = "No matching version of the data found in DVC."
        raise ValueError(msg)
