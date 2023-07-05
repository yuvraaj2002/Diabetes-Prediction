import sys
from dataclasses import dataclass
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import save_object
import numpy as np


@dataclass
class Data_processsing_configuration:
    processor_path = os.path.join("artifacts", "processor.pkl")


class process_data:
    def __init__(self):
        self.processor_path_variable = Data_processsing_configuration()

    # Method for building a pipeline
    def build_pipeline(self):
        try:
            # Column transformer for dealing with outliers
            yeo_transformation = ColumnTransformer(
                transformers=[
                    ("Yeo-Johnson", PowerTransformer(), [0, 1, 2, 3, 4, 5, 6, 7])
                ],
                remainder="passthrough",
            )

            # Column transformer to do feature scaling
            scaling_transformer = ColumnTransformer(
                transformers=[
                    ("scale_transformer", MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7])
                ],
                remainder="passthrough",
            )

            # Let's build a pipeline
            pipe = Pipeline(
                steps=[
                    ("Yeo-Johnson-Transformation", yeo_transformation),
                    ("Scaling", scaling_transformer),
                ]
            )
            logging.info("Pipeline created succesfully")
            return pipe

        except Exception as e:
            raise CustomException(e, sys)

    def intialize_data_processing(self, train_path):
        try:
            logging.info("Intializing data transformation")

            # Let's now fetch the training data
            train_data = pd.read_csv(train_path)

            # Seperating out the input features and target variable
            input_features = train_data[train_data.columns[:-1]].values
            output_variable = train_data[train_data.columns[-1]].values
            logging.info("Separated the data into dependent and indpendent data")

            # Let's processing the data
            pipeline_object = self.build_pipeline()
            train_data = pipeline_object.fit_transform(train_data)

            train_arr = np.c_[train_data, np.array(output_variable)]

            save_object(file_path=self.processor_path_variable.processor_path, obj=pipeline_object)


            return (train_arr, self.processor_path_variable.processor_path)

        except Exception as e:
            raise CustomException(e, sys)
