import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import yaml
from dvc.repo import Repo as DvcRepo
from git import Repo as GitRepo
from pyprojroot import here

from dmd_era5 import (
    config_reader,
    create_mock_era5,
    log_and_print,
    setup_logger,
    slice_era5_dataset,
    thin_era5_dataset,
)

config = config_reader("era5-download")
logger = setup_logger("ERA5Download", "era5_download.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def validate_time_parameters(parsed_config: dict) -> None:
    """
    Validate the time-related parameters in the from the user config.

    Args:
        parsed_config (dict): The parsed configuration dictionary.

    Raises:
        ValueError: If any of the time parameters are invalid or inconsistent.
    """

    start_datetime = parsed_config["start_datetime"]
    end_datetime = parsed_config["end_datetime"]
    delta_time = parsed_config["delta_time"]

    # Check if end datetime is after start datetime
    if end_datetime <= start_datetime:
        msg = "End datetime must be after start datetime"
        raise ValueError(msg)

    # Check if the time range is at least as long as delta_time
    if (end_datetime - start_datetime) < delta_time:
        msg = f"""Time range must be at least as long as delta_time.
        {end_datetime} - {start_datetime} < {delta_time}"""
        raise ValueError(msg)

    # Check if delta_time is positive
    if delta_time <= timedelta(0):
        msg = "delta_time must be positive."
        raise ValueError(msg)

    # Check if start_datetime is not in the future
    if start_datetime > datetime.now():
        msg = "Start date cannot be in the future."
        raise ValueError(msg)


def config_parser(config: dict = config) -> dict:
    """
    Parse the configuration dictionary and return a dictionary object.

    Args:
        config (dict): Configuration dictionary with the configuration parameters.

    Returns:
        dict: Dictionary with the parsed configuration parameters.
    """

    parsed_config = {}

    # Validate the required fields
    required_fields = [
        "source_path",
        "start_datetime",
        "end_datetime",
        "delta_time",
        "variables",
        "levels",
    ]

    for field in required_fields:
        if field not in config:
            msg = f"Missing required field in config: {field}"
            logger.error(msg)
            raise ValueError(msg)

    # ------------ Parse the source path ------------
    parsed_config["source_path"] = config["source_path"]

    # ------------ Parse the start date and time ------------
    try:
        parsed_config["start_datetime"] = datetime.fromisoformat(
            config["start_datetime"]
        )
    except ValueError as e:
        msg = f"Invalid start datetime format in config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # ------------ Parse the delta time ------------
    delta_time_mapping = {
        "h": lambda x: timedelta(hours=int(x)),
        "d": lambda x: timedelta(days=int(x)),
        "w": lambda x: timedelta(weeks=int(x)),
        "m": lambda x: timedelta(days=int(x) * 365 // 12),
        "y": lambda x: timedelta(days=int(x) * 365),
    }

    try:
        # Get the unit of the delta time
        unit = config["delta_time"][-1].lower()

        # Get the number of units
        num_units = int(config["delta_time"][:-1])

        if unit in delta_time_mapping:
            parsed_config["delta_time"] = delta_time_mapping[unit](num_units)
        else:
            msg = f"Unsupported delta_time format in config: {config['delta_time']}"
            logger.error(msg)
            raise ValueError(msg)

    except ValueError as e:
        msg = f"Error parsing delta_time from config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # ------------ Parse the end date and time ------------
    try:
        parsed_config["end_datetime"] = datetime.fromisoformat(config["end_datetime"])
    except ValueError as e:
        msg = f"Invalid end datetime format in config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # Validate the time parameters
    validate_time_parameters(parsed_config)

    # ------------ Parse variables ------------
    try:
        if config["variables"] == "all":
            parsed_config["variables"] = ["all"]
        else:
            parsed_config["variables"] = [
                v.strip() for v in config["variables"].split(",")
            ]
    except ValueError as e:
        msg = f"Error parsing variables from config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # ------------ Parse levels ------------
    try:
        parsed_config["levels"] = [int(level) for level in config["levels"].split(",")]
    except ValueError as e:
        msg = f"Error parsing levels from config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # ------------ Generate  the save path ------------
    # The file will be saved with the following name format:
    # - "{start_datetime}_{end_datetime}_{delta_time}.nc"
    # in the `data/era5_download` directory

    start_str = parsed_config["start_datetime"].strftime("%Y-%m-%dT%H")
    end_str = parsed_config["end_datetime"].strftime("%Y-%m-%dT%H")
    delta_str = config["delta_time"]
    parsed_config["save_name"] = f"{start_str}_{end_str}_{delta_str}.nc"
    parsed_config["save_path"] = os.path.join(
        here(), "data/era5_download", parsed_config["save_name"]
    )

    return parsed_config


def add_config_attributes(ds: xr.Dataset, parsed_config: dict) -> xr.Dataset:
    """
    Add the configuration settings as attributes to the ERA5 dataset.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        parsed_config (dict): The parsed configuration dictionary.

    Returns:
        xr.Dataset: The ERA5 dataset with the configuration settings as attributes.
    """
    ds.attrs["source_path"] = parsed_config["source_path"]
    ds.attrs["start_datetime"] = parsed_config["start_datetime"].isoformat()
    ds.attrs["end_datetime"] = parsed_config["end_datetime"].isoformat()
    ds.attrs["hours_delta_time"] = parsed_config["delta_time"].total_seconds() / 3600
    ds.attrs["variables"] = parsed_config["variables"]
    ds.attrs["levels"] = parsed_config["levels"]
    ds.attrs["date_downloaded"] = datetime.now().isoformat()
    return ds


def add_config_to_dvc_log(
    dvc_file_path: str, parsed_config: dict, git_add=False
) -> None:
    """
    Add the configuration settings as metadata to a custom log file.
    Each entry in the log file stores metadata for a single download
    under a unique DVC md5 hash.

    Args:
        dvc_file_path (str): The path to the DVC file.
        parsed_config (dict): The parsed configuration dictionary.
        git_add (bool): Whether to stage the log file for commit.
    """

    # get the md5 hash of the dvc file
    with open(dvc_file_path) as f:
        dvc_file_content = yaml.safe_load(f)
    md5_hash = dvc_file_content["outs"][0]["md5"]

    log_file = parsed_config["save_name"].split(".")[0]
    log_file = os.path.join(here(), "data/era5_download", f"{log_file}.yaml")

    # Create the log file if it does not exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    # Add the metadata to the log file
    with open(log_file, "a") as f:
        f.write(f"{md5_hash}:\n")
        f.write(f"  source_path: {parsed_config['source_path']}\n")
        f.write(f"  start_datetime: {parsed_config['start_datetime'].isoformat()}\n")
        f.write(f"  end_datetime: {parsed_config['end_datetime'].isoformat()}\n")
        f.write(
            f"  hours_delta_time: {parsed_config['delta_time'].total_seconds()/3600}\n"
        )
        f.write(f"  variables: {parsed_config['variables']}\n")
        f.write(f"  levels: {parsed_config['levels']}\n")
        f.write(f"  date_downloaded: {datetime.now().isoformat()}\n")

    # Stage the log file for commit
    if git_add:
        with GitRepo(here()) as repo:
            repo.index.add([log_file])


def download_era5_data(parsed_config: dict, use_mock_data: bool = False) -> xr.Dataset:
    """
    Download ERA5 data from the specified source path and return an xarray Dataset.

    Args:
        parsed_config (dict): Parsed configuration dictionary with the
        configuration parameters.

    Returns:
        xr.Dataset: An xarray Dataset containing the downloaded ERA5 data.
    """

    try:
        if use_mock_data:
            log_and_print(logger, "Creating mock ERA5 data...")
            full_era5_ds = create_mock_era5(
                start_datetime=parsed_config["start_datetime"],
                end_datetime=parsed_config["end_datetime"],
                variables=parsed_config["variables"]
                if parsed_config["variables"] != ["all"]
                else ["temperature", "u_component_of_wind", "v_component_of_wind"],
                levels=parsed_config["levels"],
            )

            # Override source_path for mock data
            parsed_config["source_path"] = "mock_data"

            log_and_print(logger, "Mock ERA5 data created.")

        else:
            # Open the ERA5 Dataset
            log_and_print(logger, "Loading ERA5 Dataset...")
            full_era5_ds = xr.open_dataset(parsed_config["source_path"], engine="zarr")
            log_and_print(logger, "ERA5 loaded.")

            # Select the variables
            if parsed_config["variables"] != ["all"]:
                full_era5_ds = full_era5_ds[parsed_config["variables"]]

        # Selecting the time range and levels
        log_and_print(logger, "Slicing ERA5 Dataset...")
        era5_ds = slice_era5_dataset(
            full_era5_ds,
            parsed_config["start_datetime"],
            parsed_config["end_datetime"],
            parsed_config["levels"],
        )

        # Apply time thinning if delta_time is greater than 1 hour
        if parsed_config["delta_time"] > timedelta(hours=1):
            log_and_print(logger, "Thinning ERA5 Dataset...")
            era5_ds = thin_era5_dataset(era5_ds, parsed_config["delta_time"])

        # Add config settings as attributes to the dataset
        era5_ds = add_config_attributes(era5_ds, parsed_config)

        # Save the dataset as NetCDF
        if not use_mock_data:
            log_and_print(
                logger, f"Size of ERA5 Dataset: {np.round(era5_ds.nbytes / 1e6)} MB"
            )
            log_and_print(
                logger, f"Saving ERA5 Dataset to {parsed_config['save_path']}..."
            )
            era5_ds.to_netcdf(parsed_config["save_path"], format="NETCDF4")
            log_and_print(logger, "ERA5 Dataset saved.")

        return era5_ds

    except Exception as e:
        msg = f"""
        Error {'creating mock' if use_mock_data else 'downloading'} ERA5 Dataset: {e}
        """
        log_and_print(logger, msg, level="error")
        raise ValueError(msg) from e


def main(use_mock_data: bool = False, add_to_dvc: bool = False) -> None:
    """Main function to run the ERA5 download process."""
    try:
        parsed_config = config_parser()
        download_era5_data(parsed_config, use_mock_data)
        log_and_print(logger, "ERA5 download process completed successfully.")
    except ValueError as e:
        log_and_print(logger, f"Configuration error: {e}", level="error")
    except Exception as e:
        log_and_print(logger, f"ERA5 download process failed: {e}", level="error")

    if add_to_dvc:
        try:
            log_and_print(logger, "Adding data to DVC...")
            with DvcRepo(here()) as repo:
                repo.add(parsed_config["save_path"])
            dvc_file_path = os.path.join(parsed_config["save_path"] + ".dvc")
            add_config_to_dvc_log(dvc_file_path, parsed_config, git_add=True)
            log_and_print(logger, "Data added to DVC.")
        except Exception as e:
            log_and_print(logger, f"Error adding data to DVC: {e}", level="error")


if __name__ == "__main__":

    def check_if_dvc_repo():
        """Check if the current directory is a DVC repository."""
        try:
            with DvcRepo(here()) as _:
                return True
        except Exception:
            return False

    is_dvc_repo = check_if_dvc_repo()
    if not is_dvc_repo:
        log_and_print(
            logger, "Not a DVC repository. Not adding data to DVC.", level="warning"
        )
        log_and_print(
            logger, "To initialize a DVC repository, run `dvc init`.", level="warning"
        )
        main()
    else:
        main(add_to_dvc=True)
