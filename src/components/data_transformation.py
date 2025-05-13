import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        Returns a ColumnTransformer for preprocessing
        '''
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data for transformation.")

            # Target column
            target_column = 'YIELD'

            # Define feature columns - ensure these match your actual data
            numerical_columns = [
                'Year', 'CONDITION', 'PROGRESS', 'PRICE RECEIVED', 'STOCKS', 
                'SALES', 'Moisture', 'sand_per', 'slit_per', 'clay_per', 'ph', 
                'Cation Exchange Capacity', 'Organic Matter', 
                'Available Water Capacity', 'ksat', 'slope', 'elev'
            ]
            categorical_columns = [
                'Period', 'Geo Level', 'State', 'Commodity', 
                'Soil_Texture', 'Soil_Fertility'
            ]

            # Validate columns exist in data
            numerical_columns = [col for col in numerical_columns if col in train_df.columns]
            categorical_columns = [col for col in categorical_columns if col in train_df.columns]

            # Separate features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column].values.reshape(-1, 1)  # Ensure 2D array
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column].values.reshape(-1, 1)     # Ensure 2D array

            # Clean column names
            X_train.columns = X_train.columns.str.strip()
            X_test.columns = X_test.columns.str.strip()

            # Create and fit preprocessor
            logging.info("Creating preprocessing object")
            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)
            
            logging.info("Applying preprocessing to training and test data")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Combine features and targets
            train_arr = np.hstack((X_train_processed, y_train))
            test_arr = np.hstack((X_test_processed, y_test))

            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")
            logging.info("Data transformation completed successfully")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)