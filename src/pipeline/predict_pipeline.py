import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Get the absolute path to the artifacts directory
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)), 
                                     "artifacts", "model.pkl"))
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                           "artifacts", "preprocessor.pkl")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Ensure column names match training data
            features.columns = features.columns.str.strip()
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)