import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
) 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting Training and testing input data.")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Logistic regression": LogisticRegression(),
                "KNeighbour classifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "Random forest classifier": RandomForestClassifier(),
                "XGB classifier": XGBClassifier(),
                "Adaboost classifier": AdaBoostClassifier()
            }

            model_report:dict =  evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                    raise Exception("No best model found.")
            
            logging.info(f"Best model on both training and testing dataset.")

            save_object(
                 file_path = self.model_trainer_config.trained_file_path,
                 obj = best_model
            )

            predicted = best_model.predict(X_test)

            class_report = classification_report(y_test, predicted)

            return class_report
        
        except Exception as e:
            raise CustomException(e, sys)