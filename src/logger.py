"""Logging setup for the prediction system."""
import logging
import sys
from config import LOGGING_CONFIG

def setup_logger(name):
    """Setup and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG['level'])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING_CONFIG['level'])
    
    # File handler
    file_handler = logging.FileHandler(LOGGING_CONFIG['log_file'])
    file_handler.setLevel(LOGGING_CONFIG['level'])
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
