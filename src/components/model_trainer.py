"""
This module contains a class for training a machine learning model using
ElasticNet regression. The class reads in configuration files, prepares the model
with specified hyperparameters, trains the model on a given dataset, and
saves the trained model.
"""

from os.path import dirname, normpath

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.constants import CONFIGS, PARAMS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import (
    create_directories,
    read_yaml,
    remove_key,
    save_as_json,
    save_as_pickle,
)
from src.utils.model_utils import evaluate_classification_models, get_best_model


class ModelTrainer:
    """
    A class used to train a machine learning model using linear regression.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_trainer

        # Read the model params
        self.params = read_yaml(PARAMS)

        # Input file path
        self.train_array_path = normpath(self.configs.train_array_path)
        self.test_array_path = normpath(self.configs.test_array_path)

        # Output file path
        self.model_path = normpath(self.configs.model_path)
        self.scores_path = normpath(self.configs.scores_path)

    def evaluate_models(self) -> list[dict]:
        """
        summary
        """
        # Load the training and test set array
        train_array = np.load(self.train_array_path)
        test_array = np.load(self.test_array_path)

        # Split train_array into features and target
        x_train, y_train = train_array[:, :-1], train_array[:, -1]
        x_test, y_test = test_array[:, :-1], test_array[:, -1]

        # Log the train shapes
        logger.info("The shape of x_train: %s", x_train.shape)
        logger.info("The shape of y_train: %s", y_train.shape)

        # Log the test shapes
        logger.info("The shape of x_test: %s", x_test.shape)
        logger.info("The shape of y_test: %s", y_test.shape)

        # Models to apply
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
        }

        # Model hyperparameters
        hyperparameters = {
            "Decision Tree": self.params.decision_tree.to_dict(),
            "Random Forest": self.params.random_forest.to_dict(),
        }

        # Evaluate models
        logger.info("Evaluating models with grid search")
        model_scores = evaluate_classification_models(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            models=models,
            params=hyperparameters,
            binary_classification=True,
        )
        logger.info("Model evaluation completed.")
        return model_scores

    def train_best_model(self):
        """
        summary
        """
        try:
            # Create directories if not exists
            create_directories([dirname(self.model_path), dirname(self.scores_path)])

            # Get model scores
            model_scores = self.evaluate_models()

            # Save model scores
            scores_data = [remove_key(item, "model") for item in model_scores]
            save_as_json(self.scores_path, scores_data)

            # Create scores dataframe
            logger.info(
                "Getting the best model on the basis of f1_score on test dataset"
            )
            scores_df = pd.DataFrame(model_scores)
            best_model_info = get_best_model(
                scores_df, evaluation_metric="f1_score_test"
            )

            # Extract best model
            best_model = best_model_info["model"]

            # Saving the best estimator
            save_as_pickle(self.model_path, best_model)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
