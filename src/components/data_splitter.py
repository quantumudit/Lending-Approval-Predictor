"""
This module contains the DataPreparation class which is used for preparing
the training and testing datasets.
"""

from os.path import dirname, normpath

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import CONFIGS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml


class DataSplitter:
    """
    This class is used for preparing the training and testing datasets. It
    reads the configuration files and prepares the datasets accordingly.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_preparation

        # Define input filepath parameters
        self.processed_filepath = normpath(self.configs.processed_data_path)

        # Define output filepath parameters
        self.train_filepath = normpath(self.configs.train_path)
        self.test_filepath = normpath(self.configs.test_path)

        # Define splitting parameters
        self.test_size = self.configs.test_size_pct
        self.random_seed = self.configs.random_seed

    def prepare_train_test_sets(self):
        """
        This function prepares the training and testing datasets. It creates
        the necessary directories if they do not exist and saves the datasets
        in the specified file paths. If there is an error during the process,
        it raises a CustomException.

        Raises:
            CustomException: If there is an error during the preparation
            of the datasets.
        """
        try:
            # Create directory if not exist
            create_directories(
                [
                    dirname(self.train_filepath),
                    dirname(self.test_filepath),
                ]
            )

            # Read the raw dataset
            customers_df = pd.read_csv(self.processed_filepath)

            # Prepare training and test datasets
            train_set, test_set = train_test_split(
                customers_df, test_size=self.test_size, random_state=self.random_seed
            )

            # Save the training datasets
            train_set.to_csv(
                self.train_filepath, index=False, header=True, encoding="utf-8"
            )
            logger.info("Training data saved at: %s", self.train_filepath)
            logger.info("Train set shape: %s", train_set.shape)

            # Save the training datasets
            test_set.to_csv(
                self.test_filepath, index=False, header=True, encoding="utf-8"
            )
            logger.info("Test data saved at: %s", self.test_filepath)
            logger.info("Test set shape: %s", test_set.shape)

        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
