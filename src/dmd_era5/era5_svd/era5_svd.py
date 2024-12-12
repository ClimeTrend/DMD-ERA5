import logging
import os
import sys
from datetime import datetime

import numpy as np
import xarray as xr
from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]

from dmd_era5 import retrieve_data_from_dvc
from dmd_era5.core import (
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


def add_config_attributes(ds: xr.Dataset, parsed_config: dict) -> xr.Dataset:
    """
    Add the configuration settings as attributes to the SVD results dataset.

    Args:
        ds (xr.Dataset): The SVD results dataset.
        parsed_config (dict): The parsed configuration dictionary.

    Returns:
        xr.Dataset: The SVD results dataset with the
        configuration settings as attributes.
    """

    ds.attrs["source_path"] = parsed_config["source_path"]
    ds.attrs["n_components"] = parsed_config["n_components"]
    ds.attrs["variables"] = parsed_config["variables"]
    ds.attrs["levels"] = parsed_config["levels"]
    ds.attrs["mean_center"] = int(parsed_config["mean_center"])
    ds.attrs["scale"] = int(parsed_config["scale"])
    ds.attrs["delay_embedding"] = parsed_config["delay_embedding"]
    ds.attrs["svd_type"] = parsed_config["svd_type"]
    ds.attrs["era5_slice_path"] = parsed_config["era5_slice_path"]
    ds.attrs["date_processed"] = datetime.now().isoformat()
    return ds


def retrieve_era5_slice(
    parsed_config: dict, use_dvc: bool = False
) -> tuple[xr.Dataset, bool]:
    """
    Given the configuration parameters, retrieve a slice of ERA5 data
    from the working directory or DVC.

    Args:
        parsed_config (dict): The parsed configuration dictionary.
        use_dvc (bool): Whether to use Data Version Control (DVC).

    Returns:
        xr.Dataset: The ERA5 slice.
        bool: Whether the ERA5 slice was retrieved from DVC.
    """

    retrieved_from_dvc = False

    def ensure_list(obj: str | list) -> list[str]:
        return [obj] if isinstance(obj, str) else obj

    def check_era5_slice(era5_ds: xr.Dataset) -> bool:
        era5_ds_attrs = era5_ds.attrs
        return (
            sorted(parsed_config["variables"])
            == sorted(
                set(ensure_list(era5_ds_attrs["variables"]))
                & set(parsed_config["variables"])
            )
            and sorted(parsed_config["levels"])
            == sorted(
                set(era5_ds_attrs["levels"].tolist()) & set(parsed_config["levels"])
            )
            and parsed_config["source_path"] == era5_ds_attrs["source_path"]
        )

    def retrieve_from_dvc() -> xr.Dataset:
        log_and_print(logger, "Attempting to retrieve ERA5 slice from DVC...")
        retrieve_data_from_dvc(parsed_config, data_type="era5_slice")
        log_and_print(
            logger, f"ERA5 slice retrieved from DVC: {parsed_config['era5_slice_path']}"
        )
        return xr.open_dataset(parsed_config["era5_slice_path"])

    if os.path.exists(parsed_config["era5_slice_path"]):
        log_and_print(logger, "ERA5 slice found in working directory.")
        era5_ds = xr.open_dataset(parsed_config["era5_slice_path"])
        if check_era5_slice(era5_ds):
            log_and_print(logger, "ERA5 slice matches configuration.")
            return era5_ds, retrieved_from_dvc
        log_and_print(logger, "ERA5 slice does not match configuration.")
        if use_dvc:
            era5_ds = retrieve_from_dvc()
            retrieved_from_dvc = True
            return era5_ds, retrieved_from_dvc
        msg = "ERA5 slice in working directory does not match configuration."
        raise ValueError(msg)
    log_and_print(logger, "ERA5 slice not found in working directory.")
    if use_dvc:
        era5_ds = retrieve_from_dvc()
        retrieved_from_dvc = True
        return era5_ds, retrieved_from_dvc
    msg = "ERA5 slice not found in working directory."
    raise FileNotFoundError(msg)


def retrieve_svd_results(
    parsed_config: dict, use_dvc: bool = False
) -> tuple[xr.Dataset, bool]:
    """
    Given the configuration parameters, retrieve SVD results from the working directory
    or DVC.

    Args:
        parsed_config (dict): The parsed configuration dictionary.
        use_dvc (bool): Whether to use Data Version Control (DVC).

    Returns:
        xr.Dataset: The SVD results.
        bool: Whether the SVD results were retrieved from DVC.
    """

    retrieved_from_dvc = False

    def ensure_list(obj: str | list) -> list[str]:
        return [obj] if isinstance(obj, str) else obj

    def check_svd_results(svd_ds: xr.Dataset) -> bool:
        svd_ds_attrs = svd_ds.attrs
        return (
            parsed_config["source_path"] == svd_ds_attrs["source_path"]
            and parsed_config["n_components"] == svd_ds_attrs["n_components"]
            and parsed_config["variables"] == ensure_list(svd_ds_attrs["variables"])
            and parsed_config["levels"] == svd_ds_attrs["levels"].tolist()
            and parsed_config["mean_center"] == svd_ds_attrs["mean_center"]
            and parsed_config["scale"] == svd_ds_attrs["scale"]
            and parsed_config["delay_embedding"] == svd_ds_attrs["delay_embedding"]
        )

    def retrieve_from_dvc() -> xr.Dataset:
        log_and_print(logger, "Attempting to retrieve SVD results from DVC...")
        retrieve_data_from_dvc(parsed_config, data_type="era5_svd")
        log_and_print(
            logger, f"SVD results retrieved from DVC: {parsed_config['svd_path']}"
        )
        return xr.open_dataset(parsed_config["svd_path"])

    if os.path.exists(parsed_config["svd_path"]):
        log_and_print(logger, "SVD results found in working directory.")
        svd_ds = xr.open_dataset(parsed_config["svd_path"])
        if check_svd_results(svd_ds):
            log_and_print(logger, "SVD results match configuration.")
            return svd_ds, retrieved_from_dvc
        log_and_print(logger, "SVD results do not match configuration.")
        if use_dvc:
            svd_ds = retrieve_from_dvc()
            retrieved_from_dvc = True
            return svd_ds, retrieved_from_dvc
        msg = "SVD results in working directory do not match configuration."
        raise ValueError(msg)
    log_and_print(logger, "SVD results not found in working directory.")
    if use_dvc:
        svd_ds = retrieve_from_dvc()
        retrieved_from_dvc = True
        return svd_ds, retrieved_from_dvc
    msg = "SVD results not found in working directory."
    raise FileNotFoundError(msg)


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

    Returns:
        xr.Dataset: The xarray Dataset containing the SVD results.
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
    return (False, False)


if __name__ == "__main__":
    main()
