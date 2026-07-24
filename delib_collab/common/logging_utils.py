#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import logging
from logging.handlers import RotatingFileHandler
from colorlog import ColoredFormatter
from datetime import datetime
import yaml
from delib_collab.paths import PROJECT_ROOT

DEFAULT_LOG_ROOT = PROJECT_ROOT / 'logs'

class YamlFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'name': record.name,
            'line_number': record.lineno,
            'module': record.module,
            'function': record.funcName
        }
        return yaml.dump(log_obj, default_flow_style=False, allow_unicode=True)

def setup_logger(name, log_level=logging.DEBUG, log_folder=None):
    """
    Set up a named logger with file and console handlers.
    :param name: Logger name for distinguishing different loggers
    :param log_level: Logging level, default DEBUG
    :param log_folder: Log folder path, defaults to PROJECT_ROOT/logs
    :return: Configured logger instance
    """
    if log_folder is None:
        log_folder = str(DEFAULT_LOG_ROOT)

    if not os.path.isabs(log_folder):
        log_folder = os.path.join(str(DEFAULT_LOG_ROOT), log_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    log_file = os.path.join(log_folder, f'{name}_{current_time}.log')

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
