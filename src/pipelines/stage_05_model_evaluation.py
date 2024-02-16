"""WIP
"""

from src.components.model_evaluation import ModelEvaluation
from src.exception import CustomException
from src.logger import logger


class ModelEvaluationPipeline:
    """_summary_"""

    def __init__(self):
        pass

    def main(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            logger.info("Model Evaluation started")
            model_eval = ModelEvaluation()
            model_eval.save_evaluation_results()
            logger.info("Model evaluation completed successfully")
        except Exception as excp:
            logger.error(CustomException(excp))
            raise CustomException(excp) from excp


if __name__ == "__main__":
    STAGE_NAME = "Model Evaluation Stage"

    try:
        logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e
