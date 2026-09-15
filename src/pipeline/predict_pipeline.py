import os
import sys
import pandas as pd

from src.exceptions import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:

            print("Before loading model and preprocessor")

            model = load_object(
                file_path=self.model_path
            )

            preprocessor = load_object(
                file_path=self.preprocessor_path
            )

            print("After loading model and preprocessor")

            data_scaled = preprocessor.transform(features)

            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
        Age,
        Smoking,
        Alcohol_Use,
        Obesity,
        Diet_Red_Meat,
        Diet_Salted_Processed,
        Physical_Activity,
        Air_Pollution,
        Occupational_Hazards,
        Calcium_Intake,
        BMI,
        Physical_Activity_Level,
        Gender,
        Family_History,
        BRCA_Mutation,
        H_Pylori_Infection
    ):
        self.Age = Age
        self.Smoking = Smoking
        self.Alcohol_Use = Alcohol_Use
        self.Obesity = Obesity
        self.Diet_Red_Meat = Diet_Red_Meat
        self.Diet_Salted_Processed = Diet_Salted_Processed
        self.Physical_Activity = Physical_Activity
        self.Air_Pollution = Air_Pollution
        self.Occupational_Hazards = Occupational_Hazards
        self.Calcium_Intake = Calcium_Intake
        self.BMI = BMI
        self.Physical_Activity_Level = Physical_Activity_Level

        self.Gender = Gender
        self.Family_History = Family_History
        self.BRCA_Mutation = BRCA_Mutation
        self.H_Pylori_Infection = H_Pylori_Infection

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],

                "Smoking": [self.Smoking],

                "Alcohol_Use": [self.Alcohol_Use],

                "Obesity": [self.Obesity],

                "Diet_Red_Meat": [self.Diet_Red_Meat],

                "Diet_Salted_Processed": [
                    self.Diet_Salted_Processed
                ],

                "Physical_Activity": [
                    self.Physical_Activity
                ],

                "Air_Pollution": [
                    self.Air_Pollution
                ],

                "Occupational_Hazards": [
                    self.Occupational_Hazards
                ],

                "Calcium_Intake": [
                    self.Calcium_Intake
                ],

                "BMI": [self.BMI],

                "Physical_Activity_Level": [
                    self.Physical_Activity_Level
                ],

                "Gender": [self.Gender],

                "Family_History": [
                    self.Family_History
                ],

                "BRCA_Mutation": [
                    self.BRCA_Mutation
                ],

                "H_Pylori_Infection": [
                    self.H_Pylori_Infection
                ]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)














