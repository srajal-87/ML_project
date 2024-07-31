'''
-> This module is responsible for preprocessing the data before it is fed into the machine learning models.
-> It defines pipelines for transforming numerical and categorical features (e.g., scaling, encoding).
-> The preprocessed data is then saved for later use. This module also uses logging to report the status of the transformations and any issues encountered
'''

import os
import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

# Define a data class for the data transformation configuration
@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object as a pickle file
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Define the DataTransformation class
class DataTransformation:
    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating and returning a preprocessor object
        for transforming the dataset.
        """
        try:
            # Define the numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create a pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("scaler", StandardScaler())  # Standardize numerical features
                ]
            )

            # Create a pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # One-hot encode categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Standardize categorical features without centering
                ]
            )

            # Log the categorical and numerical columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline
                    ("cat_pipeline", cat_pipeline, categorical_columns)   # Apply categorical pipeline
                ]
            )

            return preprocessor  # Return the preprocessor object
        
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for performing data transformation on the
        training and testing data.
        """
        try:
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name and input features for training
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit the preprocessor on the training data and transform both training and testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the input features and target feature into training and testing arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed training and testing arrays along with the preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
