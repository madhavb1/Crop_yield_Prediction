import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self, train_path: str = None, test_path: str = None):
        """
        Runs the complete training pipeline
        
        Args:
            train_path (str, optional): Path to training data
            test_path (str, optional): Path to test data
        """
        try:
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
                train_path=train_path, 
                test_path=test_path
            )
            
            # Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, 
                test_data_path
            )
            
            # Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Training completed with R2 score: {r2_score}")
            return r2_score
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e