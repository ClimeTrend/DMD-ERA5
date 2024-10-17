"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

from .config_reader import config_reader
from .logger import setup_logger, log_and_print


__all__ = ["config_reader", "setup_logger", "log_and_print"]
__version__ = version(__name__)
