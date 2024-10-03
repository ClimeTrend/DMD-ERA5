from configparser import ConfigParser
from pyprojroot import here
import os

# Load the config file
config_path =  os.path.join(here(),"src/dmd_era5/config.ini")


# Reads the config file and returns a dictionary
def config():
    parser = ConfigParser()
    parser.read(config_path, encoding="utf-8-sig")
    
    config_dict = {}

    # Loop through sections in the config file
    for section in parser.sections():
        # Add section and its variables to the dictionary
        config_dict[section] = dict(parser.items(section))
    return config_dict


# ---- Helper Functions ----

# Reads the config parser and returns the section
def section_reader(parser):
    section_list = []
    for section in parser.sections():
        section_list.append(parser[section])
    return section_list

# Reads settings for a specific section and returns them
def settings_reader(section):
    if parser.has_section(section):
         # Dictionary of key-value pairs for the section
        return dict(parser.items(section)) 
    else:
        return {}
    

config_data = config()
print(config_data)