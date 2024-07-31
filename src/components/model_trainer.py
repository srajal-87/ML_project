'''
-> This module is responsible for training various machine learning models on the preprocessed data.
-> It evaluates different models using techniques like GridSearchCV to find the best hyperparameters.
-> The performance of each model is recorded, and the best-performing model is saved for future predictions.
'''

import os  #Provides functions to interact with the operating system.
import sys  #Provides access to system-specific parameters and functions.
from dataclasses import dataclass # A decorator to automatically generate special methods like __init__ and __repr__ for classes.

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Define a data class for the model trainer configuration
@dataclass
class ModelTrainerConfig:
    # Path to save the trained model as a pickle file
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Define the ModelTrainer class
class ModelTrainer:
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            # Split the train and test arrays into input features (X) and target feature (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last one for training data
                train_array[:, -1],   # The last column for training data
                test_array[:, :-1],   # All columns except the last one for testing data
                test_array[:, -1]     # The last column for testing data
            )

            # Define a dictionary of regression models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models using the custom evaluate_models function
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the evaluation report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Check if the best model's score is above the threshold (0.6)
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Use the best model to make predictions on the test set
            predicted = best_model.predict(X_test)

            # Calculate the R² score to evaluate the model's performance on the test set
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
