import os
from datetime import datetime, timedelta
from logging import Logger


def validate_time_parameters(parsed_config: dict) -> None:
    """
    Validate the time-related parameters in the from the user config.

    Args:
        parsed_config (dict): The parsed configuration dictionary.

    Raises:
        ValueError: If any of the time parameters are invalid or inconsistent.
    """

    start_datetime = parsed_config["start_datetime"]
    end_datetime = parsed_config["end_datetime"]
    delta_time = parsed_config["delta_time"]

    # Check if end datetime is after start datetime
    if end_datetime <= start_datetime:
        msg = "End datetime must be after start datetime"
        raise ValueError(msg)

    # Check if the time range is at least as long as delta_time
    if (end_datetime - start_datetime) < delta_time:
        msg = f"""Time range must be at least as long as delta_time.
        {end_datetime} - {start_datetime} < {delta_time}"""
        raise ValueError(msg)

    # Check if delta_time is positive
    if delta_time <= timedelta(0):
        msg = "delta_time must be positive."
        raise ValueError(msg)

    # Check if start_datetime is not in the future
    if start_datetime > datetime.now():
        msg = "Start date cannot be in the future."
        raise ValueError(msg)


def config_parser(config: dict, section: str, logger: Logger | None = None) -> dict:
    """
    Parse the configuration dictionary and return a dictionary object.

    Args:
        config (dict): Configuration dictionary with the configuration parameters.
        section (str): Section of the configuration file.
        logger (Logger): Logger object for logging messages. Default is None.

    Returns:
        dict: Dictionary with the parsed configuration parameters.
    """

    parsed_config = {}

    if section == "era5-download":
        required_fields = [
            "source_path",
            "start_datetime",
            "end_datetime",
            "delta_time",
            "variables",
            "levels",
        ]
        save_folder = "data/era5_download"
    else:
        msg = f"Section {section} not currently supported."
        raise ValueError(msg)

    # Check for required fields
    for field in required_fields:
        if field not in config:
            msg = f"Missing required field in config: {field}"
            if logger is not None:
                logger.error(msg)
            raise ValueError(msg)

    # Parse the source path
    parsed_config["source_path"] = config["source_path"]

    # Parse the start and end datetimes
    try:
        parsed_config["start_datetime"] = datetime.fromisoformat(
            config["start_datetime"]
        )
        parsed_config["end_datetime"] = datetime.fromisoformat(config["end_datetime"])
    except ValueError as e:
        msg = f"Invalid datetime format in config: {e}"
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg) from e

    # Parse the delta time
    delta_time_mapping = {
        "h": lambda x: timedelta(hours=int(x)),
        "d": lambda x: timedelta(days=int(x)),
        "w": lambda x: timedelta(weeks=int(x)),
        "m": lambda x: timedelta(days=int(x) * 365 // 12),
        "y": lambda x: timedelta(days=int(x) * 365),
    }
    try:
        # Get the unit of the delta time
        unit = config["delta_time"][-1].lower()
        # Get the number of units
        num_units = int(config["delta_time"][:-1])
        if unit in delta_time_mapping:
            parsed_config["delta_time"] = delta_time_mapping[unit](num_units)
        else:
            msg = f"Unsupported delta_time format in config: {config['delta_time']}"
            if logger is not None:
                logger.error(msg)
            raise ValueError(msg)
    except ValueError as e:
        msg = f"Error parsing delta_time from config: {e}"
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg) from e

    # Validate the time parameters
    validate_time_parameters(parsed_config)

    # Parse the variables
    try:
        if config["variables"] == "all":
            parsed_config["variables"] = ["all"]
        else:
            parsed_config["variables"] = [
                v.strip() for v in config["variables"].split(",")
            ]
    except ValueError as e:
        msg = f"Error parsing variables from config: {e}"
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg) from e

    # Parse the levels
    try:
        if config["levels"] == "all":
            parsed_config["levels"] = ["all"]
        else:
            parsed_config["levels"] = [
                int(level) for level in config["levels"].split(",")
            ]
    except ValueError as e:
        msg = f"Error parsing levels from config: {e}"
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg) from e

    # The file will be saved with the following name format:
    # - "{start_datetime}_{end_datetime}_{delta_time}.nc"
    start_str = parsed_config["start_datetime"].strftime("%Y-%m-%dT%H")
    end_str = parsed_config["end_datetime"].strftime("%Y-%m-%dT%H")
    delta_str = config["delta_time"]
    parsed_config["save_name"] = f"{start_str}_{end_str}_{delta_str}.nc"
    parsed_config["save_path"] = os.path.join(save_folder, parsed_config["save_name"])

    return parsed_config
