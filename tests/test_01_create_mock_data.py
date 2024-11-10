"""
Tests for the create_mock_data module.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from dmd_era5.create_mock_data import create_mock_era5


def test_create_mock_era5():
    """Test that the create_mock_era5 function correctly creates a mock ERA5 dataset."""
    mock_ds = create_mock_era5(
        start_datetime="2019-01-01",
        end_datetime="2019-01-02",
        variables=["temperature"],
        levels=[1000, 850, 500],
    )

    assert isinstance(mock_ds, xr.Dataset)
    assert "temperature" in mock_ds.data_vars
    assert list(mock_ds.level.values) == [1000, 850, 500]

