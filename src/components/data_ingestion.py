'''
-> This module is responsible for collecting and loading the data from the source (e.g., CSV files).
-> It reads the data into a DataFrame, performs train-test splits, and saves the raw, training, and test data to specified paths.
-> Logging is used to track the data ingestion process and capture any errors that may occur.
'''
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# Define a data class for the data ingestion configuration
@dataclass
class DataIngestionConfig:
    # Paths to save the train, test, and raw data
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Define the DataIngestion class
class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function is responsible for data ingestion
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Print the current working directory
            print("Current Working Directory:", os.getcwd())
            
            # Read the dataset into a DataFrame
            df = pd.read_csv(r'..\..\notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            
            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths to the train and test sets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)

# If this script is run directly, initiate the data ingestion process
if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    # Start the data ingestion process and get the train and test data paths
    train_data, test_data = obj.initiate_data_ingestion()

    # Import DataTransformation class for transforming the data
    from src.components.data_transformation import DataTransformation
    data_transformation = DataTransformation()
    # Start the data transformation process
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Import ModelTrainer class for training the model
    from src.components.model_trainer import ModelTrainer
    modeltrainer = ModelTrainer()
    # Start the model training process and print the evaluation score
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
