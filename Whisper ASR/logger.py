"""
Training logger module for managing and formatting training logs.

This module provides a configurable logger with both console and rotating file handlers
for tracking machine learning training progress. It includes specific methods for
logging training resumption and various log levels.
"""

import logging
from logging.handlers import RotatingFileHandler


class TrainingLogger:
    """
    A logger utility class for managing and formatting training logs.
    
    This class provides a unified logging interface with both console and file outputs,
    including automatic log rotation to prevent excessive file sizes. It's specifically
    designed for machine learning training workflows.

    Attributes:
        logger (logging.Logger): The logger instance for ModularTrainer.

    Methods:
        info(message): Logs an informational message.
        warning(message): Logs a warning message.
        error(message): Logs an error message.
        debug(message): Logs a debug message.
        log_training_resume(epoch, global_step, total_epochs): Logs a message indicating training has resumed.
    """

    def __init__(self, 
                 log_path: str = './logs/training.log', 
                 level: int = logging.INFO,
                 max_log_size: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        """
        Initializes the TrainingLogger with file-based and console-based logging.
        
        Creates a logger that outputs to both console and a rotating file handler.
        The file handler automatically rotates logs when they reach the specified size,
        maintaining a specified number of backup files.

        Args:
            log_path (str): Path to the log file. Directory will be created if it doesn't exist.
                Default is './logs/training.log'.
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
                Default is logging.INFO.
            max_log_size (int): Maximum size (in bytes) of a single log file before rotation.
                Default is 10 MB (10 * 1024 * 1024 bytes).
            backup_count (int): Number of backup log files to retain during rotation.
                When this number is exceeded, the oldest log file is deleted.
                Default is 5.
        """
        self.logger = logging.getLogger('ModularTrainer')
        self.logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        
        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Rotating file handler setup
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_log_size, 
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Adding handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """
        Logs an informational message.
        
        Use this for general information about training progress,
        such as epoch completions, metric updates, and normal operations.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Logs a warning message.
        
        Use this for potentially problematic situations that don't
        prevent training from continuing, such as missing optional
        configurations or deprecated features.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        Logs an error message.
        
        Use this for error conditions that may impact training results
        but don't cause the program to crash, such as failed checkpoint
        saves or data loading issues.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)
    
    def debug(self, message: str):
        """
        Logs a debug message.
        
        Use this for detailed diagnostic information useful during
        development and troubleshooting, such as tensor shapes,
        gradient values, or internal state information.

        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)
    
    def log_training_resume(self, 
                            epoch: int, 
                            global_step: int, 
                            total_epochs: int):
        """
        Logs a message indicating that training has resumed.
        
        Creates a formatted message specifically for training resumption,
        providing key information about the current training state.

        Args:
            epoch (int): The current epoch at which training is resuming (1-indexed).
            global_step (int): The global step count when training is resuming.
                This represents the total number of training steps completed.
            total_epochs (int): The total number of epochs planned for the training process.
        """
        resume_message = (
            f"Training Resumed:\n"
            f"   Current Epoch: {epoch}\n"
            f"   Global Step: {global_step}\n"
            f"   Total Epochs: {total_epochs}\n"
            f"   Remaining Epochs: {total_epochs - epoch}"
        )
        self.info(resume_message)