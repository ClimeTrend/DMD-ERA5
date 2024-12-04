import logging
import sys

import xarray as xr

from dmd_era5 import config_reader, log_and_print, setup_logger, slice_era5_dataset

config = config_reader("era5-svd")
logger = setup_logger("ERA5-SVD", "era5_svd.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def svd_on_era5(parsed_config: dict, mock_era5: xr.Dataset | None = None):
    """
    Perform Singular Value Decomposition (SVD) on the ERA5 data.

    Args:
        parsed_config (dict): Parsed configuration dictionary with
            the configuration parameters.
        mock_era5 (xarray.Dataset): Mock ERA5 data for testing purposes.

    Returns:
    """

    if mock_era5 is None:
        try:
            log_and_print(
                logger, f"Opening ERA5 file: {parsed_config['file_path']} ..."
            )
            era5_data = xr.open_dataset(parsed_config["file_path"])
        except Exception as e:
            msg = f"Error opening requested ERA5 file: {e}"
            log_and_print(logger, msg, level="error")
            raise ValueError(msg) from e
    else:
        era5_data = mock_era5

    try:
        log_and_print(logger, "Slicing ERA5 data...")
        era5_data = slice_era5_dataset(
            era5_data, parsed_config["start_datetime"], parsed_config["end_datetime"]
        )
    except Exception as e:
        msg = f"Error slicing ERA5 data: {e}"
        log_and_print(logger, msg, level="error")
        raise Exception(msg) from e


def main(
    config: dict = config, use_mock_data: bool = False, use_dvc: bool = False
) -> tuple[bool, bool]:
    """
    Main function to perform Singular Value Decomposition (SVD) on a slice of ERA5 data.

    If using DVC, the function will attempt to retrieve SVD results from DVC first,
    before performing a new SVD operation.
    If appropriate SVD results are not found, the function will attempt to retrieve an
    appropriate ERA5 slice from DVC on which to perform the SVD operation, if it cannot
    find one in the working directory.
    If an appropriate slice is not found in the working directory or DVC, an error
    will be raised.

    Args:
        config (dict): Configuration dictionary with the configuration parameters,
        optional and primarily intended for testing.
        use_mock_data (bool): Use mock data for testing purposes.
        use_dvc (bool): Whether to use Data Version Control (DVC).

    Returns:
        tuple[bool, bool]: A tuple of two booleans indicating whether the SVD results
        were added to DVC and whether they were retrieved from DVC.
    """