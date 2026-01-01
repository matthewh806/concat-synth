import logging
import sys

def setup_logger(log_level: int = logging.INFO):
    
    # Root logger
    logger = logging.getLogger("concatenative")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()


    # Console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger Initialised. Console level: {logging.getLevelName(log_level)}")