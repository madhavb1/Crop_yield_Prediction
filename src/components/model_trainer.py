import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models with consistent naming
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            # Parameters must use EXACTLY the same names as models dictionary
            params = {
                "RandomForestRegressor": {
                    'n_estimators': [8, 16],
                    'max_depth': [5, 10 ],
                    'min_samples_split': [2, 5]
                },
                "LinearRegression": {},
                "XGBRegressor": {
                    'learning_rate': [.01, .05, ],
                    'n_estimators': [32, 64,]
                },
               
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 > 0.6")
            
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Final model R2 score on test data: {r2_square}")
            return r2_square

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)