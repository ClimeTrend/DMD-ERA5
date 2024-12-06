import logging
import sys

import numpy as np
import xarray as xr
from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]

from dmd_era5.core import (
    config_parser,
    config_reader,
    log_and_print,
    setup_logger,
)

config = config_reader("era5-svd")
logger = setup_logger("ERA5-SVD", "era5_svd.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def svd_on_era5(
    da: xr.DataArray, parsed_config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) on the pre-processed ERA5 slice.

    Args:
        da (xr.DataArray): The pre-processed ERA5 slice.
        parsed_config (dict): The parsed configuration dictionary.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The SVD results U, s, and V,
        where U and V are the left and right singular vectors, respectively,
        and s is the singular values. U has shape (n_samples, n_components),
        s has shape (n_components,), and V has shape (n_components, n_features).
    """
    X = da.values
    svd_type = parsed_config["svd_type"]
    n_components = parsed_config["n_components"]
    if svd_type == "standard":
        log_and_print(logger, "Performing standard SVD...")
        U, s, V = np.linalg.svd(X, full_matrices=False)
        U = U[:, :n_components]
        s = s[:n_components]
        V = V[:n_components, :]
        log_and_print(logger, "Standard SVD complete.")
    elif parsed_config["svd_type"] == "randomized":
        log_and_print(logger, "Performing randomized SVD...")
        U, s, V = randomized_svd(X, n_components=n_components)
        log_and_print(logger, "Randomized SVD complete.")
    else:
        msg = f"SVD type {svd_type} is not supported."
        raise ValueError(msg)
    return U, s, V


def combine_svd_results(
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    coords: xr.Coordinates,
    attrs: dict | None = None,
) -> xr.Dataset:
    """
    Given the SVD results U, s, and V, combine them into an xarray Dataset.

    Args:
        U (np.ndarray): The left singular vectors.
        s (np.ndarray): The singular values.
        V (np.ndarray): The right singular vectors.
        coords (xr.Coordinates): The coordinates of the pre-processed ERA5 slice on
            which the SVD was performed.
        attrs (dict): The attributes to be added to the xarray Dataset.
    """

    U_da = xr.DataArray(
        U,
        dims=("space", "components"),
        coords={
            "space": coords["space"],
            "components": np.arange(U.shape[1]),
            "original_variable": ("space", coords["original_variable"].data),
            "delay": ("space", coords["delay"].data),
        },
    )
    s_da = xr.DataArray(
        s,
        dims=("components"),
        coords={
            "components": np.arange(s.shape[0]),
        },
    )
    V_da = xr.DataArray(
        V,
        dims=("components", "time"),
        coords={
            "components": np.arange(V.shape[0]),
            "time": coords["time"],
        },
    )
    return xr.Dataset(
        {
            "U": U_da,
            "s": s_da,
            "V": V_da,
        },
        coords=coords,
        attrs=attrs,
    )


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
    added_to_dvc = False
    retrieved_from_dvc = False

    try:
        parsed_config = config_parser(config, section="era5-svd", logger=logger)

        if use_dvc:
            pass
        else:
            pass
    except Exception as e:
        log_and_print(logger, f"ERA5 SVD process failed: {e}", level="error")

    return added_to_dvc, retrieved_from_dvc
