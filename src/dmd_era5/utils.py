from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr


def slice_era5_dataset(
    ds: xr.Dataset,
    start_datetime: datetime | str | None = None,
    end_datetime: datetime | str | None = None,
    levels: list | None = None,
) -> xr.Dataset:
    """
    Slice the ERA5 dataset based on time range and pressure levels.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        start_datetime (datetime.datetime or str): The start datetime for slicing.
            If None, the first datetime in the dataset is used. If a string, must be
            in isoformat (e.g. "2020-01-01T06").
        end_datetime (datetime.datetime or str): The end datetime for slicing.
            If None, the last datetime in the dataset is used. If a string, must be
            in isoformat (e.g. "2020-01-05").
        levels (list): The pressure levels to select.
            If None, all levels are selected.

    Returns:
        xr.Dataset: The sliced ERA5 dataset.
    """
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    first_datetime = datetime.fromtimestamp(ds.time.values[0].astype(int) * 1e-9)
    last_datetime = datetime.fromtimestamp(ds.time.values[-1].astype(int) * 1e-9)
    if start_datetime is None:
        start_datetime = first_datetime
    if end_datetime is None:
        end_datetime = last_datetime
    if levels is None:
        levels = ds.level.values

    if start_datetime < first_datetime or end_datetime > last_datetime:
        msg = "Requested time is out of range."
        raise ValueError(msg)
    try:
        return ds.sel(time=slice(start_datetime, end_datetime), level=levels)
    except KeyError as e:
        msg = "Requested level is not available in the dataset."
        raise ValueError(msg) from e


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


def create_mock_era5(start_datetime, end_datetime, variables, levels):
    """
    Create a mock ERA5-like dataset for testing purposes.

    Args:
        start_datetime (str or datetime-like):
            Start datetime of the dataset, e.g. "2020-01-01T06"
        end_datetime (str or datetime-like):
            End datetime of the dataset, e.g. ""2020-01-05"
        variables (list): List of variable names
        levels (list): List of pressure levels

    Returns:
        xr.Dataset: A mock ERA5-like dataset
    """
    # Create time range
    times = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

    # Create latitude and longitude ranges (reduced resolution for testing)
    lats = np.arange(90, -90, -5)
    lons = np.arange(-180, 180, 5)

    # Create data arrays for each variable
    data_vars = {}
    for var in variables:
        if var == "temperature":
            # Temperature decreases with height, so we'll simulate that
            data = (
                np.random.rand(len(times), len(levels), len(lats), len(lons)) * 30 + 250
            )  # Random temperatures between 250K and 280K
            for i, level in enumerate(levels):
                data[:, i, :, :] -= (
                    1000 - level
                ) / 100  # Decrease temperature with height
        elif var == "u_component_of_wind" or var == "v_component_of_wind":
            data = (
                np.random.rand(len(times), len(levels), len(lats), len(lons)) * 20 - 10
            )  # Random wind between -10 and 10 m/s
        else:
            data = (
                np.random.rand(len(times), len(levels), len(lats), len(lons)) * 100
            )  # Random data for other variables

        data_vars[var] = xr.DataArray(
            data=data,
            dims=["time", "level", "latitude", "longitude"],
            coords={
                "time": times,
                "level": levels,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "units": "K"
                if var == "temperature"
                else "m/s"
                if "wind" in var
                else "unknown"
            },
        )

    # Create the dataset
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "Conventions": "CF-1.6",
            "history": "Mock ERA5 data created for testing",
            "source": "Generated mock data",
        },
    )


def standardize_data(
    data: xr.DataArray,
    dim: str = "time",
    mean_center: bool = True,
    scale: bool = True,
) -> xr.DataArray:
    """
    Standardize the input DataArray by applying mean centering and scaling
    along the specified dimension.

    Args:
        data (xr.DataArray): The input data to standardize.
        dim (str): The dimension along which to standardize. Default is "time".
        mean_center (bool): Whether to mean center the data. Default is True.
        scale (bool): Whether to scale the data. Default is True.

    Returns:
        xr.DataArray: The standardized data.
    """

    if mean_center:
        data -= data.mean(dim=dim)
    if scale:
        data /= data.std(dim=dim)
    return data
