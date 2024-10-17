import logging
import os
from pyprojroot import here

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and log file."""
    
    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler for logging to a file
    log_path = here("logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, log_file)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create a console handler for printing to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create a logger and set its level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def log_and_print(logger: logging.Logger, message: str, level: str = 'info'):
    """Log a message and print it to the console."""
    log_function = getattr(logger, level.lower())
    log_function(message)
    print(message)