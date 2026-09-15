import os
import sys

import numpy as np
import dill
import pickle

from src.exceptions import CustomException
from sklearn.metrics import classification_report, f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred, average = "weighted")
            class_report = classification_report(y_test, y_pred)

            report[list(models.keys())[i]] = f1

            print(
                f"\n{list(models.keys())[i]}\n"
                f"{class_report}"
            )
            
        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
