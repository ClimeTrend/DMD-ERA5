import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from dmd_era5.logger import log_and_print, setup_logger
from dmd_era5.slice_tools import (
    apply_delay_embedding,
    flatten_era5_variables,
    standardize_data,
)

# Set up logger
logger = setup_logger("MockData", "mock_data.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def create_mock_era5(
    start_datetime: datetime | str,
    end_datetime: datetime | str,
    variables: list[str],
    levels: list[int],
) -> xr.Dataset:
    """
    Create a mock ERA5-like dataset for testing purposes.

    Parameters
    ----------
    start_datetime : str or datetime
        Start datetime of the dataset, e.g. "2020-01-01T06"
    end_datetime : str or datetime
        End datetime of the dataset, e.g. "2020-01-05"
    variables : list of str
        List of variable names to include (e.g. ["temperature", "u_component_of_wind"])
    levels : list of int
        List of pressure levels in hPa (e.g. [1000, 850, 500])
    lat_step : float, optional
        Latitude grid spacing in degrees, defaults to 5.0
    lon_step : float, optional
        Longitude grid spacing in degrees, defaults to 5.0

    Returns
    -------
    xr.Dataset
        A mock ERA5-like dataset with realistic spatial and temporal structure

    Examples
    --------
    mock_ds = create_mock_era5(
        start_datetime="2020-01-01",
        end_datetime="2020-01-02",
        variables=["temperature"],
        levels=[1000, 850, 500]
    )
    """
    # Create time range
    times = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

    # Create latitude and longitude ranges (reduced resolution for testing)
    lat_step = 5.0
    lon_step = 5.0
    lats = np.arange(90, -90, -lat_step)
    lons = np.arange(-180, 180, lon_step)

    # Create data arrays for each variable
    data_vars = {}
    for var in variables:
        # Generate fake data
        data = _generate_variable_data(var, times, levels, lats, lons)

        # Create the data array
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

    log_and_print(logger, "Mock ERA5 dataset creation completed successfully")

    # Create the dataset
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "Conventions": "CF-1.6",
            "history": "Mock ERA5 data created for testing",
            "source": "Generated mock data",
        },
    )


def _generate_variable_data(
    var_name: str,
    times: pd.DatetimeIndex,
    levels: list[int],
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Generate realistic mock data for a specific variable.

    Parameters
    ----------
    var_name : str
        Name of the variable to generate data for
    times, levels, lats, lons : array-like
        Coordinate arrays for the data

    Returns
    -------
    np.ndarray
        4D array of mock data with shape (time, level, lat, lon)
    """
    shape = (len(times), len(levels), len(lats), len(lons))

    if var_name == "temperature":
        # Temperature decreases with height and latitude
        data = np.random.rand(*shape) * 30 + 250  # Base temperature

        # Add vertical structure
        for i, level in enumerate(levels):
            data[:, i, :, :] -= (1000 - level) / 100

        # Add latitudinal structure
        lat_factor = np.cos(np.radians(lats))
        data = data * lat_factor[np.newaxis, np.newaxis, :, np.newaxis]

    elif "wind" in var_name:
        # Wind components with realistic magnitudes
        data = np.random.rand(*shape) * 20 - 10

    else:
        # Generic random data for other variables
        data = np.random.rand(*shape) * 100

    return data


def create_mock_era5_svd(
    start_datetime: datetime | str = "2020-01-01",
    end_datetime: datetime | str = "2020-01-02",
    variables: list[str] | None = None,
    levels: list[int] | None = None,
    mean_center: bool = True,
    scale: bool = False,
    delay_embedding: int = 2,
    n_components: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, xr.Coordinates, xr.DataArray]:
    """
    Create a mock dataset of SVD results on ERA5 for testing purposes.

    Parameters
    ----------
    start_datetime : str or datetime
        Start datetime of the dataset
    end_datetime : str or datetime
        End datetime of the dataset
    variables : list of str
        List of variable names to include. Defaults to ["temperature"].
    levels : list of int
        List of pressure levels. Defaults to [1000].
    mean_center : bool
        Whether to mean-center the data before SVD
    scale : bool
        Whether to scale the data before SVD
    delay_embedding : int
        Number of time lags to include in the delay embedding
    n_components : int
        Number of SVD components to retain

    Returns
    -------
    tuple of np.ndarray, xr.Coordinates and xr.DataArray
        Tuple of U, s, V, and coordinates of the data on which SVD was performed,
        and the original data array.
    """
    # Create the mock ERA5 dataset
    if variables is None:
        variables = ["temperature"]
    if levels is None:
        levels = [1000]
    ds = create_mock_era5(start_datetime, end_datetime, variables, levels)
    if mean_center:
        ds, _, _ = standardize_data(ds, scale=scale)
    da = flatten_era5_variables(ds)
    da = apply_delay_embedding(da, delay_embedding)

    # Perform SVD
    U, s, V = np.linalg.svd(da.values, full_matrices=False)
    U = U[:, :n_components]
    s = s[:n_components]
    V = V[:n_components, :]

    return U, s, V, da.coords, da
