import ast
import os
from configparser import ConfigParser
from pyprojroot import here

# Define the config_path variable
CONFIG_PATH = os.path.join(here(), "src/dmd_era5/config.ini")

def config_reader(section: str, config_path: str = CONFIG_PATH) -> dict:
    """
    Read the configuration file and return a dictionary object.

    Args:
        section (str): Section of the configuration file.
        config_path (str): Path to the configuration file.
            Default is src/dmd_era5/config.ini.

    Returns:
        dict: Dictionary with the configuration parameters.

    Raises:
        Exception: If the section is not found in the configuration file.
    """

    # Create parser for the configuration file
    parser = ConfigParser()
    parser.read(config_path, encoding="utf-8-sig")

    config_dict = {}

    # Check if the section exists in the configuration file
    if parser.has_section(section):
        parameters = parser.items(section)  # returns a list of item name and value
        
        for param_name, param_value in parameters:
            try:
                # Use ast.literal_eval to safely evaluate the parameter value
                config_dict[param_name] = ast.literal_eval(param_value)
            except Exception as e:
                print(
                    f"""
                    Error while parsing {param_name} from {section} section
                    in the config file: {e}
                    """
                )
                raise
    else:
        msg = f"Section {section} not found in the {config_path} file"
        raise Exception(msg)

    return config_dict