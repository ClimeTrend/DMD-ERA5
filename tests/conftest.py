import pytest


@pytest.fixture(scope="module")
def era5_download_base_config():
    return {
        "source_path": "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "start_datetime": "2019-01-01T00",
        "end_datetime": "2019-01-01T04",
        "delta_time": "1h",
        "variables": "all",
        "levels": "1000",
    }


@pytest.fixture(scope="module")
def era5_download_config_a(era5_download_base_config):
    config = era5_download_base_config.copy()
    config["variables"] = "temperature"
    config["levels"] = "1000"
    return config


@pytest.fixture(scope="module")
def era5_download_config_b(era5_download_base_config):
    config = era5_download_base_config.copy()
    config["variables"] = "u_component_of_wind"
    config["levels"] = "850"
    return config


@pytest.fixture(scope="module")
def era5_download_config_c(era5_download_base_config):
    config = era5_download_base_config.copy()
    config["variables"] = "temperature,v_component_of_wind"
    config["levels"] = "1000,925"
    return config


@pytest.fixture(scope="module")
def era5_download_config_d(era5_download_base_config):
    config = era5_download_base_config.copy()
    config["variables"] = "temperature"
    config["levels"] = "500"
    return config
