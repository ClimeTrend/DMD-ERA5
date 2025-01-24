"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

# Config file handling
from dmd_era5.config_parser import config_parser
from dmd_era5.config_reader import config_reader

# Create mock data
from dmd_era5.create_mock_data import create_mock_era5, create_mock_era5_svd

# DVC tools
from dmd_era5.dvc_tools import add_data_to_dvc, retrieve_data_from_dvc

# ERA5 download
from dmd_era5.era5_download import download_era5_data

# Logging
from dmd_era5.logger import log_and_print, setup_logger
from dmd_era5.slice_tools import (
    apply_delay_embedding,
    flatten_era5_variables,
    resample_era5_dataset,
    slice_era5_dataset,
    space_coord_to_level_lat_lon,
    standardize_data,
)

__all__ = [
    "config_parser",
    "config_reader",
    "log_and_print",
    "setup_logger",
    "create_mock_era5",
    "create_mock_era5_svd",
    "add_data_to_dvc",
    "retrieve_data_from_dvc",
    "download_era5_data",
    "apply_delay_embedding",
    "flatten_era5_variables",
    "resample_era5_dataset",
    "slice_era5_dataset",
    "space_coord_to_level_lat_lon",
    "standardize_data",
]
__version__ = version(__name__)
