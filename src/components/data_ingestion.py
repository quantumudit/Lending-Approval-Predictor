"""
This module is used for data ingestion and preprocessing. It reads raw data from a
specified path, preprocesses it, and saves the processed data to a specified path.
"""

from os.path import dirname, normpath

import pandas as pd

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml


class DataIngestion:
    """
    This class is responsible for data ingestion and preprocessing. It reads
    configuration and schema from yaml files, reads raw data from a specified path,
    preprocesses it, and saves the processed data to a specified path.
    """

    def __init__(self):
        # Read config files
        self.configs = read_yaml(CONFIGS).data_ingestion
        self.schema = read_yaml(SCHEMA).raw_data_columns

        # Input paths
        self.raw_filepath = normpath(self.configs.raw_data_path)

        # Output paths
        self.processed_filepath = normpath(self.configs.processed_data_path)

    def preprocess_data(self) -> pd.DataFrame:
        """
        Reads the raw CSV data, renames the columns according to the schema, and
        returns the preprocessed data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Read the raw CSV data
        df = pd.read_csv(self.raw_filepath)

        # Rename columns
        df = df.rename(columns=dict(self.schema))

        return df

    def save_processed_data(self) -> None:
        """
        Creates a directory if it does not exist, performs data preprocessing, and
        saves the preprocessed data as a CSV file. If an exception occurs during this
        process, it is logged and re-raised.

        Raises:
            CustomException: If an error occurs during the data preprocessing or
            saving process.
        """
        try:
            # Create directory if not exits
            create_directories([dirname(self.processed_filepath)])

            # Perform data preprocessing
            logger.info("Ingest and Preprocess data")
            customers_df = self.preprocess_data()

            # Save the dataframe as CSV file
            customers_df.to_csv(
                self.processed_filepath, index=False, header=True, encoding="utf-8"
            )
            logger.info("Preprocessed data saved at: %s", self.processed_filepath)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
