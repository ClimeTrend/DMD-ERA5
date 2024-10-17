import logging

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and log file."""
    
    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create a console handler for printing to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create a logger and set its level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_and_print(logger: logging.Logger, message: str, level: str = 'info'):
    """Log a message and print it to the console."""
    log_function = getattr(logger, level.lower())
    log_function(message)
    print(message)