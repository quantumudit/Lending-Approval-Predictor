"""
summary
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger


def evaluate_classification_models(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    models: dict,
    params: dict,
    binary_classification: bool = True,
) -> list[dict]:
    """_summary_

    Args:
        x_train (np.array): _description_
        y_train (np.array): _description_
        x_test (np.array): _description_
        y_test (np.array): _description_
        models (dict): _description_
        params (dict): _description_
        binary_classification (bool, optional): _description_. Defaults to True.

    Returns:
        list[dict]: _description_
    """
    try:
        model_scores = []

        for model_name in models:
            logger.info("Evaluating %s model", model_name)
            model = models[model_name]
            hyperparameters = params[model_name]

            # Perform Grid Search
            logger.info("Started grid search over possible hyperparameters")
            grid_search = GridSearchCV(model, hyperparameters, cv=5)
            grid_search.fit(x_train, y_train)

            # Fetch the best parameters and fit the model on training set
            logger.info("Fitting %s model with best parameters found", model_name)
            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)

            # Perform prediction over the training and test set
            logger.info("Performing prediction over the training and test data")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Evaluate the model over training set
            logger.info("Evaluating the %s model", model_name)
            accuracy_score_train = accuracy_score(y_train, y_train_pred)
            precision_score_train = (
                precision_score(y_train, y_train_pred)
                if binary_classification
                else precision_score(y_train, y_train_pred, average="weighted")
            )
            recall_score_train = (
                recall_score(y_train, y_train_pred)
                if binary_classification
                else recall_score(y_train, y_train_pred, average="weighted")
            )
            f1_score_train = (
                f1_score(y_train, y_train_pred)
                if binary_classification
                else f1_score(y_train, y_train_pred, average="weighted")
            )

            # Evaluate the model over test set
            accuracy_score_test = accuracy_score(y_test, y_test_pred)
            precision_score_test = (
                precision_score(y_test, y_test_pred)
                if binary_classification
                else precision_score(y_test, y_test_pred, average="weighted")
            )
            recall_score_test = (
                recall_score(y_test, y_test_pred)
                if binary_classification
                else recall_score(y_test, y_test_pred, average="weighted")
            )
            f1_score_test = (
                f1_score(y_test, y_test_pred)
                if binary_classification
                else f1_score(y_test, y_test_pred, average="weighted")
            )

            # Fetch the best hyperparameter values
            best_hyperparameters = grid_search.best_params_

            # Append the results into the list
            model_scores.append(
                {
                    "model_name": model_name,
                    "model": model,
                    "hyperparameters": best_hyperparameters,
                    "accuracy_score_train": accuracy_score_train,
                    "precision_score_train": precision_score_train,
                    "recall_score_train": recall_score_train,
                    "f1_score_train": f1_score_train,
                    "accuracy_score_test": accuracy_score_test,
                    "precision_score_test": precision_score_test,
                    "recall_score_test": recall_score_test,
                    "f1_score_test": f1_score_test,
                }
            )
        return model_scores
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def get_best_model(
    scores_df: pd.DataFrame, evaluation_metric: str = "f1_score_test"
) -> dict:
    """_summary_

    Args:
        scores_df (pd.DataFrame): _description_
        evaluation_metric (str, optional): _description_. Defaults to "f1_score_test".

    Returns:
        dict: _description_
    """
    best_model_row = scores_df.nlargest(1, evaluation_metric).squeeze()

    best_model_name = best_model_row["model_name"]
    best_model = best_model_row["model"]
    best_model_hyperparameters = best_model_row["hyperparameters"]

    return {
        "model_name": best_model_name,
        "model": best_model,
        "hyperparameters": best_model_hyperparameters,
    }
