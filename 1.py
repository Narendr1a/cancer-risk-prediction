with tab1:
    st.subheader("Enter Patient Information")

    with st.form("prediction_form"):

        # -------------------------
        # Numerical Features
        # -------------------------

        col1, col2 = st.columns(2)

        with col1:
            Age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30
            )

            Smoking = st.selectbox(
                "Smoking",
                ["No", "Yes"]
            )

            Alcohol_Use = st.selectbox(
                "Alcohol Use",
                ["No", "Yes"]
            )

            Obesity = st.selectbox(
                "Obesity",
                ["No", "Yes"]
            )

            Diet_Red_Meat = st.selectbox(
                "Red Meat Consumption",
                ["No", "Yes"]
            )

            Diet_Salted_Processed = st.selectbox(
                "Salted / Processed Food Consumption",
                ["No", "Yes"]
            )

        with col2:
            Physical_Activity = st.number_input(
                "Physical Activity",
                min_value=0.0,
                value=5.0
            )

            Air_Pollution = st.number_input(
                "Air Pollution Exposure",
                min_value=0.0,
                value=5.0
            )

            Occupational_Hazards = st.number_input(
                "Occupational Hazards",
                min_value=0.0,
                value=5.0
            )

            Calcium_Intake = st.number_input(
                "Calcium Intake",
                min_value=0.0,
                value=5.0
            )

            BMI = st.number_input(
                "BMI",
                min_value=0.0,
                value=22.0
            )

            Physical_Activity_Level = st.number_input(
                "Physical Activity Level",
                min_value=0.0,
                value=5.0
            )

        # -------------------------
        # Categorical Features
        # -------------------------

        st.subheader("Medical & Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            Gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            Family_History = st.selectbox(
                "Family History",
                ["No", "Yes"]
            )

        with col2:
            BRCA_Mutation = st.selectbox(
                "BRCA Mutation",
                ["No", "Yes"]
            )

            H_Pylori_Infection = st.selectbox(
                "H. Pylori Infection",
                ["No", "Yes"]
            )

        # -------------------------
        # Convert UI values to 0/1
        # -------------------------

        Smoking = 1 if Smoking == "Yes" else 0
        Alcohol_Use = 1 if Alcohol_Use == "Yes" else 0
        Obesity = 1 if Obesity == "Yes" else 0
        Diet_Red_Meat = 1 if Diet_Red_Meat == "Yes" else 0
        Diet_Salted_Processed = 1 if Diet_Salted_Processed == "Yes" else 0

        Family_History = 1 if Family_History == "Yes" else 0
        BRCA_Mutation = 1 if BRCA_Mutation == "Yes" else 0
        H_Pylori_Infection = 1 if H_Pylori_Infection == "Yes" else 0

        # IMPORTANT:
        # Change these two if your dataset uses the opposite mapping.
        Gender = 0 if Gender == "Female" else 1

        # -------------------------
        # Prediction Button
        # -------------------------

        submit = st.form_submit_button(
            "🔍 Predict Cancer Risk"
        )

        if submit:

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

                final_data = data.get_data_as_data_frame()

                predict_pipeline = PredictPipeline()

                prediction = predict_pipeline.predict(final_data)

                prediction_mapping = {
                    0: "Low",
                    1: "Medium",
                    2: "High"
                }

                result = prediction_mapping[int(prediction[0])]

                # -------------------------
                # Display Result
                # -------------------------

                st.success("Prediction completed successfully!")

                st.markdown(
                    f"""
                    ### 🎯 Predicted Risk Level

                    ## **{result}**
                    """
                )

            except Exception:
                st.error(
                    "❌ Prediction failed. Please check your input values and try again."
                )