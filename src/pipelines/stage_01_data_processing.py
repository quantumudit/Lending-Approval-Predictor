"""
This module is used to ingest data from a source, process it and save the processed
data.

It uses the DataIngestion class from the data_ingestion module and the CustomException
class from the exception module.
"""

from src.components.data_processor import DataProcessor
from src.exception import CustomException
from src.logger import logger


class DataProcessorPipeline:
    """
    This class is used to create a data ingestion pipeline.
    It has a main method which starts the data ingestion process, saves the processed
    data and handles any exceptions that occur during the process.
    """

    def __init__(self):
        pass

    def main(self):
        """
        This method starts the data ingestion process, saves the processed data and
        handles any exceptions that occur during the process.

        Raises:
            CustomException: If any error occurs during the data ingestion process,
            it is logged and a CustomException is raised.
        """
        try:
            logger.info("Data ingestion started")
            data_ingestion = DataProcessor()
            data_ingestion.save_processed_data()
            logger.info("Data ingestion completed successfully")
        except Exception as excp:
            logger.error(CustomException(excp))
            raise CustomException(excp) from excp


if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"

    try:
        logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
        obj = DataProcessorPipeline()
        obj.main()
        logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e
