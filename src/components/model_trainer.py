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

            params = {
                    "Logistic regression": {
                        "C": [0.01, 0.1, 1, 10, 100],
                        "solver": ["lbfgs", "liblinear"],
                        "max_iter": [500, 1000, 2000]
                    },

                    "KNeighbour classifier": {
                        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                        "weights": ["uniform", "distance"],
                        "metric": ["euclidean", "manhattan", "minkowski"],
                        "p": [1, 2]
                    },

                    "SVC": {
                            "C": [0.1, 1, 10, 100],
                            "kernel": ["linear", "rbf", "poly"],
                            "gamma": ["scale", "auto", 0.01, 0.1, 1]
                        },

                    "Random forest classifier": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 5, 10, 15, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2"]
                    },

                    "XGB classifier": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 4, 5, 6],
                        "min_child_weight": [1, 2, 5],
                        "subsample": [0.7, 0.8, 1.0],
                        "colsample_bytree": [0.7, 0.8, 1.0]
                    },
                    "Adaboost classifier": {
                            "n_estimators": [50, 100, 200, 300],
                            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
                    },
            }

            model_report:dict =  evaluate_model(X_train, y_train, X_test, y_test, models, params)

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