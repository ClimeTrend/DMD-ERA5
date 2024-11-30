"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

from dmd_era5.core import (
    config_parser,
    config_reader,
    log_and_print,
    setup_logger
)
from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.era5_download import download_era5_data
from dmd_era5.dvc_tools import add_data_to_dvc, retrieve_data_from_dvc
from dmd_era5.slice_tools import (
    resample_era5_dataset,
    slice_era5_dataset,
    standardize_data,
)

__all__ = [
    "config_reader",
    "setup_logger",
    "log_and_print",
    "resample_era5_dataset",
    "slice_era5_dataset",
    "create_mock_era5",
    "standardize_data",
    "config_parser",
    "add_data_to_dvc",
    "retrieve_data_from_dvc",
    "download_era5_data",
]
__version__ = version(__name__)
