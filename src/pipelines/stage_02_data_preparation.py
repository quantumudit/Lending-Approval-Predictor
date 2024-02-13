"""WIP
"""

from src.components.data_preparation import DataPreparation
from src.exception import CustomException
from src.logger import logger


class DataPreparationPipeline:
    """_summary_"""

    def __init__(self):
        pass

    def main(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            logger.info("Data Preparation started")
            data_prep = DataPreparation()
            data_prep.prepare_train_test_sets()
            logger.info("Data preparation completed successfully")
        except Exception as excp:
            logger.error(CustomException(excp))
            raise CustomException(excp) from excp


if __name__ == "__main__":
    STAGE_NAME = "Data Preparation Stage"

    try:
        logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e
