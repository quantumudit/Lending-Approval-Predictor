"""
summary
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
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


def classification_metrics(y_true, y_pred) -> dict:
    """
    Calculates the classification metrics accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    classification_dict = {
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
    }
    return classification_dict


def detailed_classification_metrics(y_true, y_pred, classes: list) -> tuple:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_

    Returns:
        tuple: _description_
    """
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_classes = (
        cr_df.iloc[: len(classes)].reset_index().rename(columns={"index": "class"})
    )
    avgs_df = cr_df.iloc[len(classes) + 1 :]
    accuracy = round(cr_dict["accuracy"], 3)
    return (cr_classes, avgs_df, accuracy)


def detailed_confusion_matrix(
    y_true, y_pred, classes: list, normalize: bool = False
) -> pd.DataFrame:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_
        normalize (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    cm = confusion_matrix(y_true, y_pred)

    cm_data = []

    # Iterate over classes and confusion matrix
    for idx, class_name in enumerate(classes):
        tp = cm[idx][idx]
        fp = sum(cm[row][idx] for row in range(len(cm)) if row != idx)
        fn = sum(cm[idx][col] for col in range(len(cm)) if col != idx)
        tn = sum(sum(row) for j, row in enumerate(cm) if j != idx) - fp
        cm_dict = {
            "class": class_name,
            "TP": tp if normalize is False else tp / (tp + fn),
            "FP": fp if normalize is False else fp / (tn + fp),
            "FN": fn if normalize is False else fn / (tp + fn),
            "TN": tn if normalize is False else tn / (tn + fp),
        }
        cm_data.append(cm_dict)
    # Construct dataframe
    cm_df = pd.DataFrame(cm_data)
    return cm_df


def get_classification_report_df(y_true, y_pred, classes: list) -> pd.DataFrame:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cr_df, _, _ = detailed_classification_metrics(y_true, y_pred, classes)
    cm_df = detailed_confusion_matrix(y_true, y_pred, classes, normalize=False)
    cm_df_norm = detailed_confusion_matrix(y_true, y_pred, classes, normalize=True)

    # Add "_norm" to cm_df_norm columns (except the "class" column)
    cm_df_norm.columns = ["class"] + [
        x + "_norm" for x in cm_df_norm.columns if x != "class"
    ]

    # Change the "class" column to "str" type
    for df in [cr_df, cm_df, cm_df_norm]:
        df["class"] = df["class"].astype(str)

    # Merge dataframes
    merged_df = cr_df.merge(cm_df, on="class").merge(cm_df_norm, on="class")
    return merged_df
