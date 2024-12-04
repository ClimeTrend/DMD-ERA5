"""
A module for constants.
"""

ERA5_PRESSURE_LEVEL_VARIABLES: set[str] = {
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
}

ERA5_SINGLE_LEVEL_VARIABLES: set[str] = {
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}

ERA5_VARIABLES = ERA5_PRESSURE_LEVEL_VARIABLES.union(ERA5_SINGLE_LEVEL_VARIABLES)


ERA5_PRESSURE_LEVELS: set[int] = {
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
}
