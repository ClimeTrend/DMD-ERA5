import logging
import sys
from datetime import datetime, timedelta

import xarray as xr

from dmd_era5 import log_and_print, setup_logger

# Set up logger
logger = setup_logger("ERA5Processing", "era5_processing.log")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def slice_era5_dataset(
    ds: xr.Dataset,
    start_datetime: datetime | str | None = None,
    end_datetime: datetime | str | None = None,
    levels: list | None = None,
) -> xr.Dataset:
    """
    Slice an ERA5 dataset by time range and pressure levels.

    Parameters
    ----------
    ds : xr.Dataset
        The input ERA5 dataset to slice.
    start_datetime : datetime or str, optional
        The start datetime for slicing. If str, must be in isoformat
        (e.g. "2020-01-01T06"). If None, uses first datetime in dataset.
    end_datetime : datetime or str, optional
        The end datetime for slicing. If str, must be in isoformat
        (e.g. "2020-01-05"). If None, uses last datetime in dataset.
    levels : list of int, optional
        The pressure levels to select. If None, selects all levels.

    Returns
    -------
    xr.Dataset
        The sliced ERA5 dataset.

    Raises
    ------
    ValueError
        If requested time range is outside dataset bounds or levels not found.
    """

    # Convert string datetimes to datetime objects if needed
    start_dt = (
        datetime.fromisoformat(start_datetime)
        if isinstance(start_datetime, str)
        else start_datetime
    )
    end_dt = (
        datetime.fromisoformat(end_datetime)
        if isinstance(end_datetime, str)
        else end_datetime
    )

    # Get dataset time bounds
    time_bounds = _get_dataset_time_bounds(ds)

    # Use dataset bounds if no times specified
    start_dt = start_dt or time_bounds["first"]
    end_dt = end_dt or time_bounds["last"]

    # Validate time range is within dataset bounds
    if start_dt < time_bounds["first"] or end_dt > time_bounds["last"]:
        msg = f"Time range ({start_dt} to {end_dt}) is outside dataset"
        msg += f"bounds ({time_bounds['first']} to {time_bounds['last']})."
        log_and_print(logger, msg, "error")
        raise ValueError(msg)

    # Validate the start is before the end datetime
    if start_dt >= end_dt:
        msg = "Start datetime must be before end datetime."
        log_and_print(logger, msg, "error")
        raise ValueError(msg)

    # Use all levels if none specified
    levels = levels or list(ds.level.values)

    # Slice the dataset
    try:
        sliced_ds = ds.sel(time=slice(start_dt, end_dt), level=levels)
        log_and_print(
            logger,
            f"Dataset slicing completed successfully using {start_dt}"
            f"to {end_dt} and levels {levels}",
        )
        return sliced_ds

    except KeyError as e:
        available_levels = list(ds.level.values)
        msg = "Requested level is not available in the dataset."
        msg += f"Available levels: {available_levels}"
        log_and_print(logger, msg, "error")
        raise ValueError(msg) from e


def _get_dataset_time_bounds(ds: xr.Dataset) -> dict:
    """
    Get the first and last timestamps from an ERA5 dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The ERA5 dataset.

    Returns
    -------
    dict
        Dictionary with 'first' and 'last' datetime objects.
    """
    return {
        "first": datetime.fromtimestamp(ds.time.values[0].astype(int) * 1e-9),
        "last": datetime.fromtimestamp(ds.time.values[-1].astype(int) * 1e-9),
    }


def thin_era5_dataset(ds: xr.Dataset, delta_time: timedelta) -> xr.Dataset:
    """
    Thin the ERA5 dataset along the time dimension based
    on the specified time delta.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        delta_time (timedelta): The time delta for thinning.

    Returns:
        xr.Dataset: The thinned ERA5 dataset.
    """

    thinned_ds = ds.resample(time=delta_time).nearest()
    log_and_print(logger, f"Thinned the dataset with time delta: {delta_time}")
    return thinned_ds


def standardize_data(
    data: xr.DataArray,
    dim: str = "time",
    scale: bool = True,
) -> xr.DataArray:
    """
    Standardize the input DataArray by applying mean centering and (optionally)
    scaling to unit variance along the specified dimension.

    Args:
        data (xr.DataArray): The input data to standardize.
        dim (str): The dimension along which to standardize. Default is "time".
        scale (bool): Whether to scale the data. Default is True.

    Returns:
        xr.DataArray: The standardized data.
    """
    log_and_print(logger, f"Standardizing data along {dim} dimension...")

    # Mean center the data
    log_and_print(
        logger,
        f"Removing mean along {dim} dimension...",
    )
    data = data - data.mean(dim=dim)
    if scale:
        # Scale the data by the standard deviation
        log_and_print(logger, f"Scaling to unit variance along {dim} dimension...")
        data = data / data.std(dim=dim)
    return data
