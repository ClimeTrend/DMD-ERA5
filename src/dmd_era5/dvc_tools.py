import os

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
