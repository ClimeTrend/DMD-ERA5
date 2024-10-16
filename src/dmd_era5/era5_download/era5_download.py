import xarray as xr

from dmd_era5.config_reader import config_reader

config = config_reader("era5-download")


def config_parser(config: dict = config) -> dict:
    """
    Parse the configuration dictionary and return a dictionary object.

    Args:
        config (dict): Configuration dictionary with the configuration parameters.

    Returns:
        dict: Dictionary with the parsed configuration parameters.
    """

    keys = [
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

    pass
