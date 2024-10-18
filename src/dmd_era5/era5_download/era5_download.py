import logging
import os
import sys
from datetime import datetime, timedelta

import xarray as xr
from pyprojroot import here

from dmd_era5 import config_reader, log_and_print, setup_logger
from dmd_era5.era5_download.create_mock_era5 import create_mock_era5

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

    start_datetime = datetime.combine(
        parsed_config["start_date"], parsed_config["start_time"]
    )
    end_datetime = datetime.combine(
        parsed_config["end_date"], parsed_config["end_time"]
    )
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

    # TODO: how to handle this?
    # Check if the time range is a multiple of delta_time
    # if (end_datetime - start_datetime) % delta_time != timedelta(0):
    #     raise ValueError(f"Time range must be a multiple of delta_time.")

    # Check if delta_time is positive
    if delta_time <= timedelta(0):
        msg = "delta_time must be positive."
        raise ValueError(msg)

    # Check if start_date is not in the future
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
        "start_date",
        "start_time",
        "end_date",
        "end_time",
        "delta_time",
        "variables",
        "levels",
        "save_name",
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
        parsed_config["start_date"] = datetime.strptime(
            config["start_date"], "%Y-%m-%d"
        )
        parsed_config["start_time"] = datetime.strptime(
            config["start_time"], "%H:%M:%S"
        ).time()
    except ValueError as e:
        msg = f"Invalid start date or time format: {e}"
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
    if (
        "end_date" not in config
        or config["end_date"] == ""
        or "end_time" not in config
        or config["end_time"] == ""
    ):
        start_datetime = datetime.combine(
            parsed_config["start_date"], parsed_config["start_time"]
        )
        parsed_config["end_date"] = start_datetime + parsed_config["delta_time"]
        parsed_config["end_time"] = parsed_config["start_time"]
        log_and_print(
            logger,
            f"""No end date/time, calculated as {parsed_config['end_date']}
            {parsed_config['end_time']} using start_date and start_time + delta_time""",
            level="warning",
        )
    else:
        try:
            parsed_config["end_date"] = datetime.strptime(
                config["end_date"], "%Y-%m-%d"
            )
            parsed_config["end_time"] = datetime.strptime(
                config["end_time"], "%H:%M:%S"
            ).time()
        except ValueError as e:
            msg = f"Invalid end time or date format in config: {e}"
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

    # ------------ Generate save_name if not provided ------------
    if not config.get("save_name"):
        # If left empty, the file will be saved with the following format:
        # - "{start_date}_{end_date}_{delta_time}.nc"

        start_str = parsed_config["start_date"].strftime("%Y-%m-%d")
        end_str = parsed_config["end_date"].strftime("%Y-%m-%d")
        delta_str = config["delta_time"]
        parsed_config["save_name"] = f"{start_str}_{end_str}_{delta_str}.nc"
    else:
        parsed_config["save_name"] = config["save_name"]

    return parsed_config


def slice_era5_dataset(
    ds: xr.Dataset, start_date: datetime, end_date: datetime, levels: list
) -> xr.Dataset:
    """
    Slice the ERA5 dataset based on time range and pressure levels.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        start_date (datetime): The start date for slicing.
        end_date (datetime): The end date for slicing.
        levels (List[int]): The pressure levels to select.

    Returns:
        xr.Dataset: The sliced ERA5 dataset.
    """
    return ds.sel(time=slice(start_date, end_date), level=levels)


def thin_era5_dataset(ds: xr.Dataset, delta_time: timedelta) -> xr.Dataset:
    """
    Thin the ERA5 dataset based on the specified time delta.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        delta_time (timedelta): The time delta for thinning.

    Returns:
        xr.Dataset: The thinned ERA5 dataset.
    """

    thinning_factor = int(delta_time.total_seconds() / 3600)
    return ds.thin(time=thinning_factor)


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
    ds.attrs["start_date"] = parsed_config["start_date"].strftime("%Y-%m-%d")
    ds.attrs["end_date"] = parsed_config["end_date"].strftime("%Y-%m-%d")
    ds.attrs["delta_time"] = parsed_config["delta_time"]
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
                start_date=parsed_config["start_date"].strftime("%Y-%m-%d"),
                end_date=parsed_config["end_date"].strftime("%Y-%m-%d"),
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
            parsed_config["start_date"],
            parsed_config["end_date"],
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
            output_path = os.path.join(here(), "data", parsed_config["save_name"])
            log_and_print(logger, f"Saving ERA5 Dataset to {output_path}...")
            era5_ds.to_netcdf(output_path, format="NETCDF4")
            log_and_print(logger, "ERA5 Dataset saved.")

        return era5_ds

    except Exception as e:
        msg = f"""Error {'creating mock' if use_mock_data else 'opening'}
        the ERA5 Dataset: {e}"""
        log_and_print(logger, msg, level="error")
        raise ValueError(msg) from e


def main(use_mock_data: bool = False) -> None:
    """Main function to run the ERA5 download process."""
    try:
        parsed_config = config_parser()
        download_era5_data(parsed_config, use_mock_data)
        log_and_print(logger, "ERA5 download process completed successfully.")
    except ValueError as e:
        log_and_print(logger, f"Configuration error: {e}", level="error")
    except Exception as e:
        log_and_print(logger, f"ERA5 download process failed: {e}", level="error")


if __name__ == "__main__":
    main()
