import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from src.logger import logging
from src.exceptions import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                            "Age",
                            "Smoking", 
                            "Alcohol_Use", 
                            "Obesity", 
                            "Diet_Red_Meat", 
                            "Diet_Salted_Processed", 
                            "Physical_Activity", 
                            "Air_Pollution", 
                            "Occupational_Hazards", 
                            "Calcium_Intake", 
                            "BMI", 
                            "Physical_Activity_Level"
                        ]
            categorical_columns = [
                            "Cancer_Type",
                            "Gender",
                            "Family_History",
                            "BRCA_Mutation",
                            "H_Pylori_Infection",
                        ]

            ordinal_features = [
                            "Risk_Level"
                        ]
            
            num_pipeline = Pipeline(
                            steps = [
                                ("imputer", SimpleImputer(strategy = "median")),
                                ("scaler", StandardScaler())
                            ]
                        )
            
            cat_pipeline = Pipeline(
                            steps = [
                                ("imputer", SimpleImputer(strategy = "most_frequent")),
                                ("One_hot_encoder", OneHotEncoder()),
                                ("scaler", StandardScaler(with_mean = False))
                            ]
                        )
            # Ordinal pipeline
            ordinal_pipeline = Pipeline(
                            steps = [
                                ('ordinal', OrdinalEncoder(categories=[['Low', 'Medium', 'High']])),
                                ('scaler', StandardScaler())
                            ]
                        )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                            [
                                ("num_pipeline", num_pipeline, numerical_columns),
                                ("cat_pipeline", cat_pipeline, categorical_columns),
                                ("ordinal_pipeline", ordinal_pipeline, ordinal_features)
                            ]
                        )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)