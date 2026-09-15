from src.pipeline.predict_pipeline import PredictPipeline, CustomData


data = CustomData(
    Age=30,
    Smoking=0,
    Alcohol_Use=0,
    Obesity=0,
    Diet_Red_Meat=0,
    Diet_Salted_Processed=0,
    Physical_Activity=2,
    Air_Pollution=1,
    Occupational_Hazards=0,
    Calcium_Intake=500,
    BMI=22.0,
    Physical_Activity_Level=2,
    Gender="Male",
    Family_History="No",
    BRCA_Mutation="No",
    H_Pylori_Infection="No"
)

df = data.get_data_as_data_frame()

print(df)

pipeline = PredictPipeline()

prediction = pipeline.predict(df)

print("Prediction:", prediction)