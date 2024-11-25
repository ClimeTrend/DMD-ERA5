import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from pyprojroot import here

from dmd_era5 import (
    config_parser,
    config_reader,
    create_mock_era5,
    log_and_print,
    resample_era5_dataset,
    setup_logger,
    slice_era5_dataset,
)

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
        if parsed_config["levels"] == ["all"]:
            parsed_config["levels"] = full_era5_ds.level.values.tolist()
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
            output_path = os.path.join(
                here(), "data/era5_download", parsed_config["save_name"]
            )
            log_and_print(
                logger, f"Size of ERA5 Dataset: {np.round(era5_ds.nbytes / 1e6)} MB"
            )
            log_and_print(logger, f"Saving ERA5 Dataset to {output_path}...")
            era5_ds.to_netcdf(output_path, format="NETCDF4")
            log_and_print(logger, "ERA5 Dataset saved.")

        return era5_ds

    except Exception as e:
        msg = f"""
        Error {'creating mock' if use_mock_data else 'downloading'} ERA5 Dataset: {e}
        """
        log_and_print(logger, msg, level="error")
        raise ValueError(msg) from e


def main(use_mock_data: bool = False) -> None:
    """Main function to run the ERA5 download process."""
    try:
        parsed_config = config_parser(
            config=config, section="era5-download", logger=logger
        )
        download_era5_data(parsed_config, use_mock_data)
        log_and_print(logger, "ERA5 download process completed successfully.")
    except ValueError as e:
        log_and_print(logger, f"Configuration error: {e}", level="error")
    except Exception as e:
        log_and_print(logger, f"ERA5 download process failed: {e}", level="error")


if __name__ == "__main__":
    main()
