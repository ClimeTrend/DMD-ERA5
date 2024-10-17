"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

from .config_reader import config_reader
from .logger import setup_logger
from .era5_download.era5_download import config_parser


__all__ = ["config_reader", "setup_logger", "config_parser"]
__version__ = version(__name__)
