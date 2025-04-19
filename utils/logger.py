# Log scripy

from datetime import datetime
import logging
import os
import sys

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H_%M_%S'
    return datetime.now().strftime(fmt)

def setup_default_logging(name, save_path, level=logging.INFO,
                          format="[%(asctime)s][%(levelname)s] - %(message)s"):

    tmp_timestr = time_str()
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=os.path.join(save_path, r'detection_{}.log'.format(tmp_timestr)),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)
    return logger, tmp_timestr
