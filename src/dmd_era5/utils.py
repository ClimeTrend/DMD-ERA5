from datetime import timedelta

import xarray as xr


def slice_era5_dataset(
    ds: xr.Dataset, start_datetime: str, end_datetime: str, levels: list
) -> xr.Dataset:
    """
    Slice the ERA5 dataset based on time range and pressure levels.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        start_datetime (str): The start datetime for slicing, e.g. '2020-01-01T00'.
        end_datetime (str): The end datetime for slicing, e.g. '2020-01-02T23'.
        levels (list): The pressure levels to select.

    Returns:
        xr.Dataset: The sliced ERA5 dataset.
    """
    return ds.sel(time=slice(start_datetime, end_datetime), level=levels)


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

    return ds.resample(time=delta_time).nearest()
