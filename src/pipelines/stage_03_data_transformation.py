"""WIP
"""

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logger


class DataTransformPipeline:
    """_summary_"""

    def __init__(self):
        pass

    def main(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            logger.info("Data Transformation started")
            data_transform = DataTransformation()
            data_transform.transform_train_test_data()
            logger.info("Data transformation completed successfully")
        except Exception as excp:
            logger.error(CustomException(excp))
            raise CustomException(excp) from excp


if __name__ == "__main__":
    STAGE_NAME = "Data Transformation Stage"

    try:
        logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
        obj = DataTransformPipeline()
        obj.main()
        logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e
