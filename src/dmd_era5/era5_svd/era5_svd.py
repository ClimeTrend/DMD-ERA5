import logging
import sys

from dmd_era5 import config_reader, setup_logger

config = config_reader("era5-svd")
logger = setup_logger("ERA5-SVD", "era5_svd.log")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def config_parser(config: dict = config) -> dict:
    """
    Parse the configuration dictionary and return the parsed dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The parsed configuration dictionary.
    """
    required_fields = [
        "file_path",
        "save_name",
        "variables",
        "levels",
        "svd_type",
        "delay_embedding",
        "standardize",
        "start_datetime",
        "end_datetime",
    ]

    for field in required_fields:
        if field not in config:
            msg = f"Missing required field in config: {field}"
            logger.error(msg)
            raise ValueError(msg)

    return {}
