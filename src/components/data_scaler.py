"""
This module provides a class for data transformation, which includes
reading configuration files, separating numerical and categorical features,
constructing a preprocessor for normalization, and transforming train and test data.

Classes:
    DataTransformation: A class for transforming data.
"""

from os.path import dirname, normpath

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, save_as_pickle


class DataScaler:
    """
    This class is responsible for transforming raw data into a format suitable for
    machine learning models. It reads configuration and schema files, constructs a
    preprocessor for normalization of numerical features, and applies this preprocessor
    to transform the train and test data. The transformed data is then saved as numpy
    arrays for further use. The class also handles the creation of necessary
    directories and the saving of the preprocessor object for future use.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_transformation
        self.schema = read_yaml(SCHEMA).processed_data_columns

        # Feature and target column names with datatype
        self.features = dict(self.schema.features)
        self.target = dict(self.schema.target)

        # Input file paths
        self.train_data_path = normpath(self.configs.train_path)
        self.test_data_path = normpath(self.configs.test_path)

        # Output file paths
        self.train_array_path = normpath(self.configs.train_array_path)
        self.test_array_path = normpath(self.configs.test_array_path)
        self.preprocessor_path = normpath(self.configs.preprocessor_path)

    def construct_preprocessor(self) -> ColumnTransformer:
        """
        Constructs a preprocessor for normalization of numerical features.

        Returns:
            Any: A preprocessor object for normalization.
        """
        # Get numerical & categorical features
        num_features = [col[0] for col in self.features.items() if col[1] != "object"]
        cat_features = [col[0] for col in self.features.items() if col[1] == "object"]

        # Pipeline to normalize numerical features
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scalar", StandardScaler()),
            ]
        )

        # Pipeline to standardize categorical features
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scalar", StandardScaler(with_mean=False)),
            ]
        )

        # Construct preprocessor object for normalization
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features),
            ]
        )
        logger.info("Preprocessor object created successfully")
        return preprocessor

    def transform_train_test_data(self) -> tuple[np.array]:
        """
        Transforms the train and test data using the constructed preprocessor.

        Returns:
            tuple[np.array]: A tuple containing two numpy arrays, one for transformed
            training data and one for transformed test data.
        """
        try:
            # Read train and test data files
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            # Get features and target
            feature_cols = list(self.features.keys())
            target_col = list(self.target.keys())

            # Spilt train & test data in-terms of features and targets
            x_train, y_train = train_df[feature_cols], train_df[target_col]
            x_test, y_test = test_df[feature_cols], test_df[target_col]

            # Log the shapes
            logger.info("The shape of X_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)
            logger.info("The shape of X_test: %s", x_test.shape)
            logger.info("The shape of y_test: %s", y_test.shape)

            # Get the preprocessor object
            preprocessor = self.construct_preprocessor()

            # Fit & transform preprocessor with the X_train data
            x_train_normalized = preprocessor.fit_transform(x_train)

            # Transform X_test with fitted preprocessor
            x_test_normalized = preprocessor.transform(x_test)

            # Convert target labels to numpy arrays
            y_train_arr = np.array(y_train.squeeze())
            y_test_arr = np.array(y_test.squeeze())

            # Create train & test arrays
            train_array = np.column_stack((x_train_normalized, y_train_arr))
            test_array = np.column_stack((x_test_normalized, y_test_arr))

            # Log the shapes
            logger.info("Shape of normalized training array: %s", train_array.shape)
            logger.info("Shape of normalized test array: %s", test_array.shape)

            # Save the arrays
            np.save(self.train_array_path, train_array)
            np.save(self.test_array_path, test_array)

            # Create directory if not exist
            create_directories([dirname(self.preprocessor_path)])

            # Saving the preprocessor object
            save_as_pickle(self.preprocessor_path, preprocessor)

            return (train_array, test_array)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
