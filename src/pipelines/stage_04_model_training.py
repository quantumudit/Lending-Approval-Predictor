"""WIP
"""

from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger


class ModelTrainerPipeline:
    """_summary_"""

    def __init__(self):
        pass

    def main(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            logger.info("Model Training started")
            model_trainer = ModelTrainer()
            model_trainer.train_best_model()
            logger.info("Model training completed successfully")
        except Exception as excp:
            logger.error(CustomException(excp))
            raise CustomException(excp) from excp


if __name__ == "__main__":
    STAGE_NAME = "Model Trainer Stage"

    try:
        logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e
