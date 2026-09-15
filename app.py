import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Title
# =========================================================

st.title("🩺 Cancer Risk Prediction System")

st.write(
    "Predict cancer risk using individual patient information "
    "or upload a CSV file for multiple predictions."
)

st.divider()


# =========================================================
# Prediction Pipeline
# =========================================================

predict_pipeline = PredictPipeline()


# =========================================================
# Tabs
# =========================================================

tab1, tab2 = st.tabs([
    "👤 Manual Information",
    "📂 CSV Prediction"
])


# =========================================================
# TAB 1 - Manual Prediction
# =========================================================

with tab1:

    st.subheader("Enter Patient Information")

    with st.form("prediction_form"):

        # -------------------------------------------------
        # Three Columns
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        # =================================================
        # Column 1
        # =================================================

        with col1:

            Age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

            Gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            Smoking = st.number_input(
                "Smoking",
                min_value=0,
                max_value=10,
                value=0
            )

            Alcohol_Use = st.number_input(
                "Alcohol Use",
                min_value=0,
                max_value=10,
                value=0
            )

            Obesity = st.number_input(
                "Obesity",
                min_value=0,
                max_value=10,
                value=0
            )

        # =================================================
        # Column 2
        # =================================================

        with col2:

            Diet_Red_Meat = st.number_input(
                "Red Meat Consumption",
                min_value=0,
                value=0
            )

            Diet_Salted_Processed = st.number_input(
                "Salted / Processed Food",
                min_value=0,
                value=0
            )

            Physical_Activity = st.number_input(
                "Physical Activity",
                min_value=0,
                value=0
            )

            Air_Pollution = st.number_input(
                "Air Pollution Exposure",
                min_value=0,
                value=0
            )

            Occupational_Hazards = st.number_input(
                "Occupational Hazards",
                min_value=0,
                value=0
            )

        # =================================================
        # Column 3
        # =================================================

        with col3:

            Calcium_Intake = st.number_input(
                "Calcium Intake",
                min_value=0,
                value=0
            )

            BMI = st.number_input(
                "BMI",
                min_value=0.0,
                value=22.0
            )

            Physical_Activity_Level = st.number_input(
                "Physical Activity Level",
                min_value=0,
                value=0
            )

            Family_History = st.selectbox(
                "Family History",
                ["Yes", "No"]
            )

            BRCA_Mutation = st.selectbox(
                "BRCA Mutation",
                ["Yes", "No"]
            )

            H_Pylori_Infection = st.selectbox(
                "H. Pylori Infection",
                ["Yes", "No"]
            )


        # =================================================
        # Convert User-Friendly Values to Dataset Format
        # =================================================

        Gender = 0 if Gender == "Female" else 1

        Family_History = 1 if Family_History == "Yes" else 0

        BRCA_Mutation = 1 if BRCA_Mutation == "Yes" else 0

        H_Pylori_Infection = 1 if H_Pylori_Infection == "Yes" else 0


        # =================================================
        # Prediction Button
        # =================================================

        submitted = st.form_submit_button(
            "🔍 Predict Risk",
            use_container_width=True
        )


    # =====================================================
    # Prediction
    # =====================================================

    if submitted:

        try:

            data = CustomData(
                Age=Age,
                Smoking=Smoking,
                Alcohol_Use=Alcohol_Use,
                Obesity=Obesity,
                Diet_Red_Meat=Diet_Red_Meat,
                Diet_Salted_Processed=Diet_Salted_Processed,
                Physical_Activity=Physical_Activity,
                Air_Pollution=Air_Pollution,
                Occupational_Hazards=Occupational_Hazards,
                Calcium_Intake=Calcium_Intake,
                BMI=BMI,
                Physical_Activity_Level=Physical_Activity_Level,
                Gender=Gender,
                Family_History=Family_History,
                BRCA_Mutation=BRCA_Mutation,
                H_Pylori_Infection=H_Pylori_Infection
            )


            # Convert input into DataFrame

            input_df = data.get_data_as_data_frame()


            # Make prediction

            prediction = predict_pipeline.predict(input_df)


            # Prediction mapping

            prediction_mapping = {
                0: "Low",
                1: "Medium",
                2: "High"
            }


            result = prediction_mapping[int(prediction[0])]


            # =================================================
            # Display Result
            # =================================================

            st.divider()

            if result == "High":

                st.error(
                    f"⚠️ Predicted Risk Level: **{result}**"
                )

            elif result == "Medium":

                st.warning(
                    f"⚠️ Predicted Risk Level: **{result}**"
                )

            else:

                st.success(
                    f"✅ Predicted Risk Level: **{result}**"
                )


        except Exception:

            st.error(
                "❌ Prediction failed. "
                "Please check your input values and try again."
            )


# =========================================================
# TAB 2 - CSV Prediction
# =========================================================

with tab2:

    st.subheader("Upload Patient CSV")

    st.write(
        "Upload a CSV containing patient information. "
        "The model will generate a risk prediction for every row."
    )


    # =====================================================
    # File Upload
    # =====================================================

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            # Read CSV

            df = pd.read_csv(uploaded_file)


            st.success(
                f"File uploaded successfully — "
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )


            # =================================================
            # Show Uploaded Data
            # =================================================

            st.subheader("Uploaded Data")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )


            # =================================================
            # Required Columns
            # =================================================

            required_columns = [

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
                "Physical_Activity_Level",
                "Gender",
                "Family_History",
                "BRCA_Mutation",
                "H_Pylori_Infection"

            ]


            # =================================================
            # Check Missing Columns
            # =================================================

            missing_columns = [

                col
                for col in required_columns
                if col not in df.columns

            ]


            if missing_columns:

                st.error(
                    "❌ Missing required columns."
                )

                st.write(missing_columns)


            else:

                # =================================================
                # Prediction Button
                # =================================================

                if st.button(
                    "🔍 Predict CSV",
                    use_container_width=True
                ):

                    try:

                        # -----------------------------------------
                        # Select Model Features
                        # -----------------------------------------

                        input_df = df[
                            required_columns
                        ].copy()


                        # -----------------------------------------
                        # Prediction
                        # -----------------------------------------

                        predictions = predict_pipeline.predict(
                            input_df
                        )


                        # -----------------------------------------
                        # Prediction Mapping
                        # -----------------------------------------

                        prediction_mapping = {
                            0: "Low",
                            1: "Medium",
                            2: "High"
                        }


                        df["Risk_Level"] = [

                            prediction_mapping[int(pred)]
                            for pred in predictions

                        ]


                        # =================================================
                        # Results
                        # =================================================

                        st.success(
                            f"Predictions completed for "
                            f"{len(df)} patients."
                        )


                        st.subheader(
                            "Prediction Results"
                        )


                        st.dataframe(
                            df,
                            use_container_width=True
                        )

                        # Risk Summary

                        st.subheader(
                            "Risk Summary"
                        )


                        risk_counts = (
                            df["Risk_Level"]
                            .value_counts()
                        )


                        col1, col2, col3 = st.columns(3)


                        with col1:

                            st.metric(
                                "Low Risk",
                                risk_counts.get("Low", 0)
                            )


                        with col2:

                            st.metric(
                                "Medium Risk",
                                risk_counts.get("Medium", 0)
                            )


                        with col3:

                            st.metric(
                                "High Risk",
                                risk_counts.get("High", 0)
                            )

                        # Download Predictions

                        csv = df.to_csv(
                            index=False
                        ).encode("utf-8")


                        st.download_button(
                            label="⬇️ Download Predictions",
                            data=csv,
                            file_name="cancer_risk_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


                    except Exception:

                        st.error(
                            "❌ Prediction failed. "
                            "Please check that the CSV contains "
                            "valid data in the required columns."
                        )


        except Exception:

            st.error(
                "❌ Could not read the uploaded CSV. "
                "Please upload a valid CSV file."
            )