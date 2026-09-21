# 🏥 Cancer Risk Prediction

## 📋 Project Overview

A comprehensive **end-to-end machine learning project** for predicting cancer risk using data science techniques. This educational project demonstrates the complete workflow of a real-world machine learning application, from data collection to model deployment.

**Course/Context:** Student Learning Project  
**Status:** ✅ Complete — Data Collection, Cleaning, EDA, Model Development & Evaluation  
**Last Updated:** September 2026

---

## 🎯 Project Objectives

This project aims to:
- Develop a predictive model for cancer risk assessment
- Demonstrate the complete machine learning pipeline
- Apply data preprocessing and exploratory data analysis (EDA) techniques
- Compare multiple machine learning algorithms
- Create a reproducible and well-documented project
- Provide insights into healthcare data analysis

---

## 📊 Project Workflow

The project follows a structured machine learning pipeline:

1. **Data Collection & Inspection** ✅
   - Loaded and explored the dataset (2,000 patients, 21 columns)
   - Confirmed no missing values, no duplicate rows
   - Documented data source and structure

2. **Data Cleaning & Preprocessing** ✅
   - Dropped non-predictive identifier (`Patient_ID`)
   - Label-encoded `Risk_Level` (Low/Medium/High → 0/1/2)
   - One-hot encoded `Cancer_Type`
   - Exported processed dataset (`cancer-risk-factors-processed.csv`)

3. **Exploratory Data Analysis (EDA)** ✅
   - Statistical summaries and distributions
   - Univariate analysis of all 16 predictors against `Risk_Level`
   - Feature correlation analysis
   - Identified patterns and a data leakage risk (see Key Insights)

4. **Feature Engineering** 🚧
   - Feature selection based on EDA findings (in progress)
   - Handling class imbalance in `Risk_Level` (planned: class weighting / SMOTE)

5. **Model Development** ✅
   - Trained and compared algorithms
   - Hyperparameter tuning
   - Cross-validation

6. **Model Evaluation** ✅
   - Accuracy 86%, Precision 0.86, Recall 0.86, F1-Score 0.86
   - Confusion matrix analysis
   - Model comparison

7. **Model Persistence** 🚧
   - Save the best-performing model
   - Document model specifications

8. **Documentation & Deployment** 🚧
   - EDA and evaluation results documented (this README)
   - Usage guides and deployment: pending

---

## 📁 Project Structure

```
cancer-risk-prediction/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned and processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   └── utils.py
├── models/
│   └── best_model.pkl        # Saved trained model
└── results/
    ├── metrics.json
    └── visualizations/
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Notebooks** | Jupyter Notebook |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Evaluation** | Scikit-learn metrics |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Narendr1a/cancer-risk-prediction.git
   cd cancer-risk-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## 📚 Usage Guide

### Running the Project

1. Start with **`01_data_exploration.ipynb`** to understand the dataset
2. Follow sequentially through the notebooks for the complete pipeline
3. Check **`notebooks/04_model_training.ipynb`** to see model development
4. Review **`notebooks/05_model_evaluation.ipynb`** for performance analysis

### Making Predictions

Once the model is trained:
```python
import pickle
import pandas as pd

# Load the model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your data
new_data = pd.DataFrame({...})

# Make predictions
predictions = model.predict(new_data)
```

---

## 📈 Expected Results

- **Model Performance Metrics**:
  - Accuracy: 86%
  - Precision: 0.86
  - Recall: 0.86
  - F1-Score: 0.86

  Given the target class imbalance (78.7% Medium baseline — see Key Insights below), confirm whether these are **macro-averaged** (treats Low/Medium/High equally) or **weighted-averaged** (dominated by the Medium class) before reporting them as-is. All four metrics landing on the identical value is a bit unusual and worth double-checking against a per-class confusion matrix, especially for the minority "High" class (102 rows) — that's the one this model actually needs to get right.

- **Key Insights** (from EDA):

  **Target distribution (`Risk_Level`)**: Medium 78.7%, Low 16.2%, High 5.1% — severely imbalanced. Requires stratified splits and class weighting / SMOTE for the minority "High" class (only 102 of 2,000 rows).

  **Strongest risk-increasing factors** (High vs. Low mean difference): Air_Pollution (+5.59), Smoking (+4.67), Alcohol_Use (+4.54), Diet_Salted_Processed (+4.15), Occupational_Hazards (+3.63), Diet_Red_Meat (+3.39), Obesity (+2.45). All show a consistent monotonic trend in their univariate distributions — risk shifts toward "High" as these values increase.

  **Protective factor**: Fruit_Veg_Intake (-1.75) — higher intake skews toward Low risk.

  **Negligible / no signal**: Age, Gender, Family_History, BRCA_Mutation, H_Pylori_Infection, Calcium_Intake, BMI, Physical_Activity_Level — flat across risk levels, not useful predictors in this dataset.

  **⚠️ Data leakage risk**: `Overall_Risk_Score` correlates 0.77 with `Risk_Level`, far above any other feature. This strongly suggests `Risk_Level` was derived (binned) from `Overall_Risk_Score`. **This column should be excluded from model features** unless proven otherwise.

  **Cancer_Type × Gender**: Breast is 455 female vs. 5 male; Prostate is 305 male vs. 0 female — expected biological pattern, but means `Gender` and `Cancer_Type` are collinear and redundant if both used to predict one another.

---

## 🔍 Dataset Information

- **Source**: `data/cancer-risk-factors.csv` (synthetic/educational cancer risk factors dataset)
- **Size**: 2,000 rows × 21 columns
- **Features**: `Cancer_Type`, `Age`, `Gender`, `Smoking`, `Alcohol_Use`, `Obesity`, `Family_History`, `Diet_Red_Meat`, `Diet_Salted_Processed`, `Fruit_Veg_Intake`, `Physical_Activity`, `Air_Pollution`, `Occupational_Hazards`, `BRCA_Mutation`, `H_Pylori_Infection`, `Calcium_Intake`, `Overall_Risk_Score`, `BMI`, `Physical_Activity_Level` (plus identifier `Patient_ID`)
- **Target Variable**: `Risk_Level` — 3-class categorical (Low / Medium / High), not binary
- **Documentation**: See `data/README.md` for detailed dataset information

---

## 📊 Language Composition

- **Jupyter Notebook**: 96.5%
- **Python**: 3.5%

---

## 🤝 Contributing

This is a student project. Suggestions and improvements are welcome!

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**IMPORTANT:** This project is for **educational and research purposes only**. It must **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult qualified healthcare professionals for medical decisions
- This model should not be deployed in clinical settings without proper validation
- Medical predictions require regulatory approval and ethical review

---

## 📝 License

This project is open source and available under the MIT License (or specify your license).

---

## 👤 Author

**Narendr1a**  
GitHub: [@Narendr1a](https://github.com/Narendr1a)

---

## 📞 Support & Questions

For questions or issues:
- Open an [Issue](https://github.com/Narendr1a/cancer-risk-prediction/issues)
- Check existing documentation
- Review the Jupyter notebooks for detailed explanations

---

## 🔗 Useful Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Jupyter Notebook Guide](https://jupyter.org/)
- [Machine Learning Best Practices](https://developers.google.com/machine-learning/crash-course)
