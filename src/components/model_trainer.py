"""
This module contains a class for training a machine learning model using
ElasticNet regression. The class reads in configuration files, prepares the model
with specified hyperparameters, trains the model on a given dataset, and
saves the trained model.
"""

from os.path import dirname, normpath

import numpy as np
from sklearn.tree import DecisionTreeClassifier, RandomForestClassifier

from src.constants import CONFIGS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, save_as_pickle


class ModelTrainer:
    """
    A class used to train a machine learning model using linear regression.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_trainer

        # Input file path
        self.train_array_path = normpath(self.configs.train_array_path)

        # Output file path
        self.model_path = normpath(self.configs.model_path)

    def train_model(self) -> LinearRegression:
        """
        Trains the linear regression model on the training dataset and
        saves the trained model.

        Returns:
            LinearRegression: The trained linear regression model.
        """
        try:
            # Load the training set array
            train_array = np.load(self.train_array_path)

            # Split train_array into features and target
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            # Log the shapes
            logger.info("The shape of x_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)
            
            # Prepare models
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
            }
            
            

            # Prepare the model
            lr_model = LinearRegression()
            logger.info("Linear regression model object initiated")

            # Fit the model on training dataset
            lr_model.fit(x_train, y_train)
            logger.info("Linear regression model fitted on training set")

            # Create directory if not exist
            create_directories([dirname(self.model_path)])

            # Saving the preprocessor object
            save_as_pickle(self.model_path, lr_model)

            return lr_model
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
