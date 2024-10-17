from datetime import datetime, timedelta
import xarray as xr

from dmd_era5.config_reader import config_reader, logger

config = config_reader("era5-download")


def config_parser(config: dict = config) -> dict:
    """
    Parse the configuration dictionary and return a dictionary object.

    Args:
        config (dict): Configuration dictionary with the configuration parameters.

    Returns:
        dict: Dictionary with the parsed configuration parameters.
    """

    parsed_config = {}

    # Validate the required fields
    required_fields = ["source_path", 
                       "start_date", 
                       "start_time", 
                       "end_date", 
                       "end_time", 
                       "delta_time", 
                       "variables", 
                       "levels", 
                       "save_name"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")


    # ------------ Parse the source path ------------
    parsed_config["source_path"] = config["source_path"]

    # ------------ Parse the start and end date ------------
    try:
        parsed_config["start_date"] = datetime.strptime(config["start_date"], "%Y-%m-%d")
        parsed_config["end_date"]   = datetime.strptime(config["end_date"], "%Y-%m-%d")
    except ValueError as e:
        msg = f"Invalid date format: {e}"
        logger.error(msg)
        raise ValueError(msg)
    
    # ------------ Parse the start and end time ------------
    try:
        parsed_config["start_time"] = datetime.strptime(config["start_time"], "%H:%M:%S").time()
        parsed_config["end_time"]   = datetime.strptime(config["end_time"], "%H:%M:%S").time()
    except ValueError as e:
        msg = f"Invalid time format in config: {e}"
        logger.error(msg)
        raise ValueError(msg)

    # ------------ Parse the delta time ------------
    delta_time_mapping = {
        "h": lambda x: timedelta(hours=int(x)),
        "d": lambda x: timedelta(days=int(x)), 
        "w": lambda x: timedelta(weeks=int(x)),
        "m": lambda x: timedelta(days=int(x) * 365 // 12),
        "y": lambda x: timedelta(days=int(x) * 365)
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
            logger.error(msg)
            raise ValueError(msg)
        
    except ValueError as e:
        msg = f"Error parsing delta_time from config: {e}"
        logger.error(msg)
        raise ValueError(msg)


    # ------------ Parse variables ------------
    try:
        if config["variables"] == "all":
            parsed_config["variables"] = ["all"]
        else:
            parsed_config["variables"] = [v.strip() for v in config["variables"].split(",")]
    except ValueError as e:
        msg = f"Error parsing variables from config: {e}"
        logger.error(msg)
        raise ValueError(msg)


    # ------------ Parse levels ------------
    try:
        parsed_config["levels"] = [int(level) for level in config["levels"].split(",")]
    except ValueError as e:
        msg = f"Error parsing levels from config: {e}"
        logger.error(msg)
        raise ValueError(msg)


    # ------------ Generate save_name if not provided ------------
    if not config.get("save_name"):
        # If left empty, the file will be saved with the following format:
        # - "{start_date}_{end_date}_{delta_time}.nc"

        start_str = parsed_config["start_date"].strftime("%Y-%m-%d")
        end_str = parsed_config["end_date"].strftime("%Y-%m-%d")
        delta_str = config["delta_time"]
        parsed_config["save_name"] = f"{start_str}_{end_str}_{delta_str}.nc"
    else:
        parsed_config["save_name"] = config["save_name"]


    return parsed_config
