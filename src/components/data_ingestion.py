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
            num_pipeline = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]

            # Categorical pipeline
            cat_pipeline = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns)
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

            # Define feature columns
            numerical_columns = [
                'Year', 'CONDITION', 'PROGRESS', 'PRICE RECEIVED', 'STOCKS', 'SALES', 'Moisture',
                'sand_per', 'slit_per', 'clay_per', 'ph', 'Cation Exchange Capacity',
                'Organic Matter', 'Available Water Capacity', 'ksat', 'slope', 'elev'
            ]
            categorical_columns = [
                'Period', 'Geo Level', 'State', 'Commodity', 'Soil_Texture', 'Soil_Fertility'
            ]

            # Drop target from features
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Clean column names (strip spaces)
            X_train.columns = X_train.columns.str.strip()
            X_test.columns = X_test.columns.str.strip()

            # Create preprocessor
            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # Fit on train, transform train and test
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Save preprocessor for inference
            with open(self.config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            # Combine features and target for model training
            train_arr = np.c_[X_train_processed, y_train.values]
            test_arr = np.c_[X_test_processed, y_test.values]

            logging.info("Data transformation completed successfully.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
