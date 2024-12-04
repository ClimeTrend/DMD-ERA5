import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from dmd_era5.core import log_and_print, setup_logger

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


def resample_era5_dataset(ds: xr.Dataset, delta_time: timedelta) -> xr.Dataset:
    """
    Resample an ERA5 dataset along the time dimension by a
    specified time delta, using nearest neighbor.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        delta_time (timedelta): The time delta for resampling.

    Returns:
        xr.Dataset: The resampled ERA5 dataset.
    """

    resampled_ds = ds.resample(time=delta_time).nearest()
    log_and_print(logger, f"Resampled the dataset with time delta: {delta_time}")
    return resampled_ds


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


def apply_delay_embedding(
    X: xr.DataArray | np.ndarray, d: int
) -> xr.DataArray | np.ndarray:
    """
    Apply delay embedding to temporal snapshots.

    Parameters
    ----------
    X : xr.DataArray or np.ndarray
        The input data array of shape (n_space * n_variables, n_time).
        If X is a DataArray, the dimensions must be ("space", "time") and
        the coordinates must include "space", "time" and "original_variable".
    d : int
        The number of snapshots from X to include in each snapshot of the output.

    Returns
    -------
    xr.DataArray or np.ndarray
        The delay-embedded data array of shape
        (n_space * n_variables * d, n_time - d + 1).
        If X is a DataArray, the output is also a DataArray with dimensions
        ("space", "time") and coordinates "space", "time", "original_variable"
        and "delay". "delay" is the delay index for each space coordinate, e.g.
        for d=2, the delay indices are [0, 0, ..., 0, 0, 1, 1, ..., 1, 1], where
        0 means no delay and 1 means a delay of 1 snapshot ahead.
        If X is a NumPy array, the output is a NumPy array.
    """

    def apply_delay_embedding_np(X, d):
        return (
            sliding_window_view(X.T, (d, X.shape[0]))[:, 0]
            .reshape(X.shape[1] - d + 1, -1)
            .T
        )

    if X.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)

    if not isinstance(d, int) or d <= 0:
        msg = "Delay must be an integer greater than 0."
        raise ValueError(msg)

    if isinstance(X, np.ndarray):
        return apply_delay_embedding_np(X, d)

    if isinstance(X, xr.DataArray):
        result = apply_delay_embedding_np(X.values, d)
        dataarray = xr.DataArray(
            result,
            dims=("space", "time"),
            coords={
                "space": np.tile(X.coords["space"], d),
                "time": X.coords["time"][: -d + 1],
                "original_variable": (
                    "space",
                    np.tile(X.coords["original_variable"], d),
                ),
                "delay": ("space", np.repeat(np.arange(d), X.coords["space"].shape[0])),
            },
            attrs=X.attrs,
        )
        dataarray.attrs["delay_embedding"] = d
        return dataarray

    msg = "Input must be a DataArray or NumPy array."
    raise ValueError(msg)


def flatten_era5_variables(era5_ds: xr.Dataset) -> xr.DataArray:
    """
    Flatten the variables in an ERA5 dataset to a single 2D array,
    returned as a DataArray with dimensions (space, time). If there is more
    than one variable in the dataset, they are stacked along the space dimension.
    In other words, the output array has shape (n_space * n_variables, n_time),
    where the first n_space elements correspond to the first variable, the next
    n_space elements correspond to the second variable, and so on.

    Parameters
    ----------
    era5_ds : xr.Dataset
        The input ERA5 dataset.

    Returns
    -------
    xr.DataArray
        The flattened array of variables, with shape (n_space * n_variables, n_time),
        where n_space = n_level * n_lat * n_lon. The DataArray has dimensions
        (space, time), and coordinates "space", "time", and "original_variable".
        "space" is a tuple of (level, latitude, longitude), and "original_variable"
        is the original variable name.
    """

    variables: list[str] = list(map(str, era5_ds.data_vars.keys()))
    coords: list[str] = list(map(str, era5_ds.coords.keys()))
    must_have_coords = ["latitude", "longitude", "time", "level"]
    spatial_stack_order = ["level", "latitude", "longitude"]

    if sorted(coords) != sorted(must_have_coords):
        msg = f"Missing required coordinates: {must_have_coords}."
        raise ValueError(msg)

    # stack the spatial dimensions
    stacked = era5_ds.stack(space=spatial_stack_order)

    # create a list of variable arrays with shape (n_space, n_time)
    data_list = [stacked[var].transpose("space", "time").values for var in variables]
    # concatenate the variable arrays along the space dimension,
    # resulting in an array of shape (n_space * n_variables, n_time)
    data_combined = np.concatenate(data_list, axis=0)

    variable_labels = np.repeat(variables, stacked.coords["space"].shape[0])

    # create a DataArray for the combined data
    dataarray = xr.DataArray(
        data_combined,
        dims=("space", "time"),
        coords={
            "space": np.tile(stacked.coords["space"], len(variables)),
            "time": stacked.coords["time"],
            "original_variable": ("space", variable_labels),
        },
        attrs=era5_ds.attrs,
    )
    dataarray.attrs["original_variables"] = variables
    dataarray.attrs["space_coords"] = spatial_stack_order

    return dataarray
