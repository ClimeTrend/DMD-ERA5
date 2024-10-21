import numpy as np
import pandas as pd
import xarray as xr


def create_mock_era5(start_datetime, end_datetime, variables, levels):
    """
    Create a mock ERA5-like dataset for testing purposes.

    Args:
        start_datetime (str):
            Start date in 'YYYY-MM-DDTHH:MM' format, e.g. '2020-01-01' or
            '2020-01-01T00'
        end_datetime (str):
            End date in 'YYYY-MM-DDTHH' format, e.g. '2020-01-02' or '2020-01-02T23'
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
