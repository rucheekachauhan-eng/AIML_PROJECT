import os
import sys
import yaml
import joblib
from typing import Any, List
from src.utils.logger import logger
from src.utils.exception import CustomException

def read_yaml(path_to_yaml: str) -> dict:
    """Reads a yaml file and returns contents as a dictionary."""
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Loaded YAML file successfully from: {path_to_yaml}")
            return content or {}
    except Exception as e:
        raise CustomException(e, sys)

def create_directories(path_to_directories: List[str], verbose: bool = True) -> None:
    """Creates a list of directories if they don't already exist."""
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: str, obj: Any) -> None:
    """Saves a python object using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Saved object to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> Any:
    """Loads a python object using joblib."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Loaded object from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
