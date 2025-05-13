import os
import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, train_path: str = None, test_path: str = None):
        """
        Reads data from provided train and test paths or uses default paths
        
        Args:
            train_path (str, optional): Path to training data. Defaults to None.
            test_path (str, optional): Path to test data. Defaults to None.
            
        Returns:
            tuple: (train_data_path, test_data_path)
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Use provided paths or default to configured paths
            train_path = train_path if train_path else self.ingestion_config.train_data_path
            test_path = test_path if test_path else self.ingestion_config.test_data_path
            
            # Read the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save datasets to artifacts
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)