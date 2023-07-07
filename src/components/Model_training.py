import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score,roc_curve, auc
from src.utils import save_object

@dataclass
class Model_training_configuration():
    model_file_path = os.path.join('artifacts','model.pkl')

class training:

    def __init__(self):
        self.file_path = Model_training_configuration()

    def initialize_training(self,X_train,y_train,test_data_path,pipeline_path):

        try:
            logging.info("Model training intiated")

            # Let's now train the model
            model = RidgeClassifier(alpha=0.16501752565697925)
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Saving the model as pickle file
            save_object(file_path=self.file_path.model_file_path, obj=model)
            logging.info("Saved the model file")

            # Let's fetch the test data
            test_data = pd.read_csv(test_data_path)
            X_test = test_data[test_data.columns[:-1]]
            y_test = test_data[test_data.columns[-1]]
            logging.info("Test data got divided into input and output")

            # Let's process the test data
            with open(pipeline_path, 'rb') as file:
                pipe = pickle.load(file)
            X_test = pipe.fit_transform(X_test)
            logging.info("Processing of the test data completed")

            # Let's get some model predictoins and compute f1 score
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            logging.info("Model is making predictoins succesfully")

            return f1

        except Exception as e:
            raise CustomException(e,sys)



