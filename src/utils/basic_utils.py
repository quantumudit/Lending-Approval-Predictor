"""
This module provides utility functions for handling files and directories.
It includes functions for reading YAML files, CSV files, creating directories
and writing to CSV files. The functions are designed to handle exceptions and
log relevant information for debugging purposes.
"""

import json
import pickle
from os import makedirs
from os.path import dirname, normpath
from typing import Any

import yaml
from box import Box
from tabulate import tabulate

from src.exception import CustomException
from src.logger import logger


def read_yaml(yaml_path: str) -> Box:
    """
    This function reads a YAML file from the provided path and returns
    its content as a Box object.

    Args:
        yaml_path (str): The path to the YAML file to be read.

    Raises:
        CustomException: If there is any error while reading the file or
        loading its content, a CustomException is raised with the original
        exception as its argument.

    Returns:
        Box: The content of the YAML file, loaded into a Box object for
        easy access and manipulation.
    """
    try:
        yaml_path = normpath(yaml_path)
        with open(yaml_path, encoding="utf-8") as yf:
            content = Box(yaml.safe_load(yf))
            logger.info("yaml file: %s loaded successfully", yaml_path)
            return content
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def create_directories(dir_paths: list, verbose=True) -> None:
    """
    This function creates directories at the specified paths.

    Args:
        dir_paths (list): A list of directory paths where directories need
        to be created.
        verbose (bool, optional): If set to True, the function will log
        a message for each directory it creates. Defaults to True.
    """
    for path in dir_paths:
        makedirs(normpath(path), exist_ok=True)
        if verbose:
            logger.info("created directory at: %s", path)


def save_as_pickle(file_path: str, model) -> None:
    """
    Save an object to a pickle file.

    Args:
        file_path (str): The file path where the pickle file will be saved.
        object (pickle): The object to be saved.

    Raises:
        CustomException: An exception occurred while saving the object.
    """
    try:
        save_path = normpath(file_path)
        with open(save_path, "wb") as pf:
            pickle.dump(model, pf)
            logger.info("pickle file: %s saved successfully", save_path)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def load_pickle(file_path: str) -> Any:
    """
    Loads a pickle file.

    Args:
        file_path (str): The file path of the pickle file.

    Returns:
        Any: The object loaded from the pickle file.
    """
    try:
        pickle_path = normpath(file_path)
        with open(pickle_path, "rb") as pf:
            loaded_object = pickle.load(pf)
            logger.info("pickle file: %s loaded successfully", pickle_path)
            return loaded_object
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def save_as_json(file_path: str, data: dict) -> None:
    """
    This function saves a dictionary as a JSON file at the specified file path.

    Args:
        file_path (str): The path where the JSON file will be saved. If the directories
        in the path do not exist, they will be created.
        data (dict): The dictionary that will be saved as a JSON file.

    Raises:
        CustomException: If there is an error during the file writing process,
        a CustomException will be raised with the original exception as its argument.
    """
    save_path = normpath(file_path)
    makedirs(dirname(save_path), exist_ok=True)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info("json file saved at: %s", save_path)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def dict_to_table(input_dict: dict, column_headers: list) -> str:
    """
    Convert a dictionary into a tabulated string.

    Args:
        input_dict (dict): The input dictionary to be converted into a table.
        column_headers (list): List of column headers for the table.

    Returns:
        str: A tabulated representation of the dictionary as a string.
    """

    table_vw = tabulate(
        input_dict.items(), headers=column_headers, tablefmt="pretty", stralign="left"
    )

    return table_vw
