import os
import os
import sys
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.components.Data_processing import process_data


@dataclass
class Data_Ingestion_Configuration:
    # Let's specify the path where we will store out train,test and validation data
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    val_path = os.path.join("artifacts", "val.csv")


class Data_Ingestion:
    def __init__(self):
        self.data_storage_paths = Data_Ingestion_Configuration()

    def intialize_data_ingestion(self):
        logging.info("Data ingestion process started")

        try:
            # Let's read the data from csv file from source directory
            df = pd.read_csv("notebook/diabetes.csv")
            logging.info("Successfully read the data from source directory")

            logging.info("Intializing the train test and validation split")
            X_train, X_test_val = train_test_split(df, test_size=0.3, random_state=0)
            X_test, X_val = train_test_split(X_test_val, test_size=0.5, random_state=0)
            logging.info("Completed train test and validation split")

            # Let's create directorires for storing the data files
            os.makedirs(
                os.path.dirname(self.data_storage_paths.train_path), exist_ok=True
            )
            logging.info("Artifacts directory created succesfully")

            X_train.to_csv(self.data_storage_paths.train_path, index=False, header=True)
            X_test.to_csv(self.data_storage_paths.test_path, index=False, header=True)
            X_val.to_csv(self.data_storage_paths.val_path, index=False, header=True)
            logging.info("Data got stored succesfully")

            return (
                self.data_storage_paths.train_path,
                self.data_storage_paths.test_path,
                self.data_storage_paths.val_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingest_obj = Data_Ingestion()
    (
        train_data_path,
        test_data_path,
        val_data_path,
    ) = data_ingest_obj.intialize_data_ingestion()

    data_process_obj = process_data()
    train_arr = data_process_obj.intialize_data_processing(train_data_path)
