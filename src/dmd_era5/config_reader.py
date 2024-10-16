import ast
import os
from configparser import ConfigParser

from pyprojroot import here

# Load the config file
config_path = os.path.join(here(), "src/dmd_era5/config.ini")


# Reads the config file and returns a dictionary
def config(section: str):
    parser = ConfigParser()
    parser.read(config_path, encoding="utf-8-sig")

    config_dict = {}

    if parser.has_section(section):
        params = parser.items(section)  # returns a list of item name and value
        for param in params:
            try:
                config_dict[param[0]] = ast.literal_eval(parser.get(section, param[0]))
            except Exception as e:
                print(
                    f"""
                    Error while parsing {param[0]} from {section} section
                    in the config file: {e}
                    """
                )
                raise
    else:
        msg = f"Section {section} not found in the {config_path} file"
        raise Exception(msg)

    return config_dict
