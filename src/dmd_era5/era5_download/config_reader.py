from configparser import ConfigParser
from pyprojroot import here
import os

# Load the config file
config_path =  os.path.join(here(),"src/dmd_era5/config.ini")

parser = ConfigParser()
parser.read(config_path, encoding="utf-8-sig")

print(parser.sections())



def config():

    return config_dict


# ---- Helper Functions ----

def section_reader():

    return section

def variable_reader():

    return variable