import logging
import sys
from datetime import datetime

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

    parsed_config = {}

    # check for required fields
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

    # parse datetime fields
    try:
        parsed_config["start_datetime"] = datetime.fromisoformat(
            config["start_datetime"]
        )
    except ValueError as e:
        msg = f"Invalid start datetime format in config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    try:
        parsed_config["end_datetime"] = datetime.fromisoformat(config["end_datetime"])
    except ValueError as e:
        msg = f"Invalid end datetime format in config: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # parse SVD type
    parsed_config["svd_type"] = config["svd_type"]
    if parsed_config["svd_type"] not in ["standard", "randomized"]:
        msg = f"Invalid SVD type in config: {parsed_config['svd_type']}"
        logger.error(msg)
        raise ValueError(msg)

    # parse delay embedding
    parsed_config["delay_embedding"] = config["delay_embedding"]
    if (
        not isinstance(parsed_config["delay_embedding"], int)
        or parsed_config["delay_embedding"] < 1
    ):
        msg = f"Invalid delay embedding in config: {parsed_config['delay_embedding']}"
        logger.error(msg)
        raise ValueError(msg)

    return parsed_config
