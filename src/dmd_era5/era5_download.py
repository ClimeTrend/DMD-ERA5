import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from dvc.repo import Repo as DvcRepo
from pyprojroot import here

from dmd_era5.core import config_parser, config_reader, log_and_print, setup_logger
from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.dvc_tools import add_data_to_dvc, retrieve_data_from_dvc
from dmd_era5.slice_tools import resample_era5_dataset, slice_era5_dataset

config = config_reader("era5-download")
logger = setup_logger("ERA5Download", "era5_download.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


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


def download_era5_data(parsed_config: dict, use_mock_data: bool = False) -> xr.Dataset:
    """
    Download ERA5 slice from the specified source path, save it to a NetCDF file,
    and return the downloaded data as an xarray Dataset.


    Args:
        parsed_config (dict): Parsed configuration dictionary with the
        configuration parameters.
        use_mock_data (bool): Whether to use mock data instead of downloading,
        for testing purposes. If True, the data will not be saved to disk.

    Returns:
        xr.Dataset: An xarray Dataset containing the downloaded ERA5 data
        or the mock data.
    """

    try:
        if use_mock_data:
            log_and_print(logger, "Creating mock ERA5 data...")
            full_era5_ds = create_mock_era5(
                start_datetime=parsed_config["start_datetime"],
                end_datetime=parsed_config["end_datetime"],
                variables=parsed_config["variables"],
                levels=parsed_config["levels"],
            )

            # Override source_path for mock data
            parsed_config["source_path"] = "mock_data"

            log_and_print(logger, "Mock ERA5 data created.")

        else:
            # Open the ERA5 Dataset
            log_and_print(logger, "Loading ERA5 Dataset...")
            full_era5_ds = xr.open_zarr(
                parsed_config["source_path"], chunks={"time": 100}
            )
            log_and_print(logger, "ERA5 loaded.")

            # Select the variables
            full_era5_ds = full_era5_ds[parsed_config["variables"]]

        # Selecting the time range and levels
        log_and_print(logger, "Slicing ERA5 Dataset...")
        era5_ds = slice_era5_dataset(
            full_era5_ds,
            parsed_config["start_datetime"],
            parsed_config["end_datetime"],
            parsed_config["levels"],
        )

        # Apply time resampling if delta_time is greater than 1 hour
        if parsed_config["delta_time"] > timedelta(hours=1):
            log_and_print(logger, "Resampling ERA5 Dataset in time...")
            era5_ds = resample_era5_dataset(era5_ds, parsed_config["delta_time"])

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


def main(
    config: dict = config, use_mock_data: bool = False, use_dvc: bool = False
) -> tuple[bool, bool]:
    """
    Main function to run the ERA5 download process.
    If using DVC, the function will attempt to retrieve the data from DVC first
    before downloading it.

    Args:
        config (dict): Configuration dictionary with the configuration parameters,
        optional and primarily for testing purposes.
        use_mock_data (bool): Whether to use mock data instead of downloading,
        for testing purposes.
        use_dvc (bool): Whether to use Data Version Control (DVC) to track the data.

    Returns:
        tuple[bool, bool]: A tuple of two booleans indicating whether the data was
        added to DVC and whether it was retrieved from DVC.
    """

    added_to_dvc = False
    retrieved_from_dvc = False
    try:
        parsed_config = config_parser(config, section="era5-download", logger=logger)

        def handle_download_and_dvc() -> bool:
            """
            Helper function to download data and add it to DVC.
            Returns True if the data was added to DVC successfully, False otherwise.
            """
            era5_ds = download_era5_data(parsed_config, use_mock_data)
            log_and_print(logger, "ERA5 download process completed successfully.")
            try:
                log_and_print(logger, "Adding ERA5 slice to DVC...")
                add_data_to_dvc(parsed_config["save_path"], era5_ds.attrs)
                log_and_print(logger, "ERA5 slice added to DVC.")
                return True
            except Exception as e:
                log_and_print(
                    logger, f"Error adding ERA5 slice to DVC: {e}", level="error"
                )
                return False

        if use_dvc:
            log_and_print(logger, "Attempting to retrieve ERA5 slice from DVC...")
            try:
                retrieve_data_from_dvc(parsed_config)
                log_and_print(
                    logger,
                    f"ERA5 slice retrieved from DVC: {parsed_config['save_path']}",
                )
                retrieved_from_dvc = True
            except (FileNotFoundError, ValueError) as e:
                log_and_print(
                    logger,
                    f"Could not retrieve ERA5 slice from DVC: {e}",
                    level="warning",
                )
                added_to_dvc = handle_download_and_dvc()
        else:
            download_era5_data(parsed_config, use_mock_data)
            log_and_print(logger, "ERA5 download process completed successfully.")

    except ValueError as e:
        log_and_print(logger, f"Configuration error: {e}", level="error")
    except Exception as e:
        log_and_print(logger, f"ERA5 download process failed: {e}", level="error")

    return added_to_dvc, retrieved_from_dvc


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
            logger,
            "Not a Data Version Control (DVC) repository. Will not use DVC.",
            level="warning",
        )
        log_and_print(
            logger, "To initialize a DVC repository, run `dvc init`.", level="warning"
        )
        main()
    else:
        main(use_dvc=True)
