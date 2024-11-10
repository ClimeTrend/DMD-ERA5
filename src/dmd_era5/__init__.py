"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

from dmd_era5.config_reader import config_reader
from dmd_era5.logger import log_and_print, setup_logger
from dmd_era5.create_mock_data import create_mock_era5
from dmd_era5.era5_processing import slice_era5_dataset, thin_era5_dataset, standardize_data

__all__ = [
    "config_reader",
    "setup_logger",
    "log_and_print",
    "thin_era5_dataset",
    "slice_era5_dataset",
    "create_mock_era5",
    "standardize_data",
]
__version__ = version(__name__)
