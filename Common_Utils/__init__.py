# common_utils/__init__.py

import logging
import os
import sys
from datetime import datetime
import time
import tracemalloc
from functools import wraps
import yaml
from pathlib import Path
from huggingface_hub import login

# ─────────────── Global Paths ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cache for loggers by filename
_logger_registry = {}


def setup_logger(name="Common_Utils", filename=None):
    """
    Sets up and returns a logger with optional filename (timestamped).

    Args:
        name (str): Logger name.
        filename (str): Base log filename (e.g. 'text_generation' — timestamp will be added).

    Returns:
        logging.Logger: Configured logger.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Default: use logger name as base for filename
    if filename is None:
        filename = f"{name}_{timestamp}.log"
    else:
        base = Path(filename).stem
        filename = f"{base}_{timestamp}.log"

    if filename in _logger_registry:
        return _logger_registry[filename]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path = LOG_DIR / filename

        # File Handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(filename)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _logger_registry[filename] = logger
    return logger



# ─── Error Utilities ─────────────────────────────
def error_message_detail(error, error_detail: sys):
    """Extracts detailed error information including file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in script: [{file_name}] at line [{exc_tb.tb_lineno}] - Message: [{str(error)}]"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# ─── Global Exception Hook ───────────────────────
def global_exception_handler(exc_type, exc_value, exc_traceback):
    logger = setup_logger()
    logger.critical("Unhandled Exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler


# ─── Decorators & Helpers ────────────────────────
def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger()
        logger.info(f"Running '{func.__name__}'...")

        start_time = time.time()
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()

        logger.info(f"'{func.__name__}' completed in {end_time - start_time:.4f} sec")
        logger.info(f"Memory used: {current / 1024:.2f} KB (peak: {peak / 1024:.2f} KB)")

        return result
    return wrapper


def load_config(path):
    logger = setup_logger()
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return cfg
    except Exception as e:
        raise CustomException(e, sys)


def upload_model_hub(cfg, model , tokenizer):
    logger = setup_logger()
    try:
        if cfg["hub"].get("upload", False):
            logger.info("Uploading model to Hugging Face Hub...")
        
        hf_token = os.getenv("HF_TOKEN") or cfg["hub"].get("token")
        if hf_token:
            login(token=hf_token)
        else:
            logger.warning("No Hugging Face token found. Skipping upload.")

        repo_id = cfg["hub"]["repo_id"]
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        logger.info(f"Model pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")

    except CustomException as e:
        logger.error(f"error occured in model_upload: {e}")

# ─── Exports ──────────────────────────────────────
__all__ = ["setup_logger", "CustomException", "track_performance", "load_config"]
