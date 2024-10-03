"""
dmd-era5: Running DMD on ERA5 data
"""

from __future__ import annotations

from importlib.metadata import version

from .era5_download.era5_download import myfunction

__all__ = ("__version__","myfunction")
__version__ = version(__name__)
