# Crop Recommendation System for Improving Crop Yield

## Project Overview
This project implements an end-to-end machine learning pipeline to:
- Classify soil nutrient levels to assess soil health.
- Predict crop yield using regression models based on soil conditions.

The goal is to support data-driven decision-making for farmers, agronomists, and agricultural planners by providing accurate predictions and soil recommendations.

---

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/madhavb1/Crop_yield_Prediction.git
cd Crop_yield_Prediction
```

### 2. Open the Project in Visual Studio Code
Use the following command in your terminal:
```
code .
```

### 3. Create and Activate Virtual Environment
```
conda create -p venv python=3.8 -y
conda activate venv/
```

### 4. Install Dependencies
```
pip install -r requirements.txt
```

### 5. Run Component Scripts
```
python -m src.components.data_ingestion
python -m src.components.data_transformation
```

### 6. Launch the Web App
```
python app.py
```
Then open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

---

## Model Performance Summary

| Task                        | Best Model              | R² / Accuracy | Key Features         |
|-----------------------------|-------------------------|------------------|----------------------|
| Soil Nutrient Classification | Random Forest Classifier | 100%             | Soil Texture         |
| Soil Nutrient Classification | XGBoost Classifier      | 83%              | Soil Fertility       |
| Crop Yield Prediction        | Random Forest Regressor | 95.4 (R²)        | Crop Yield           |
| Crop Yield Prediction        | XGBoost Regressor       | 94.5 (R²)        | Crop Yield           |
| Crop Yield Prediction        | Stacking Regressor      | 95.6 (R²)        | Combined Predictions |

---

## Folder Structure
```
Crop_yield_Prediction/
├── venv/
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   └── data_transformation.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Methodology Summary
- **Data Preprocessing**: IQR clipping, group-based imputation, encoding.
- **EDA**: Boxplots, heatmaps, correlation matrices, crop-wise analysis.
- **Modeling**:
  - Classification: Random Forest, XGBoost
  - Regression: Random Forest, XGBoost, Stacking Regressor
- **Metrics**: Accuracy, R², RMSE, MAE, cross-validation

### Model Training and Evaluation Notebooks

The `notebook/` directory contains Jupyter notebooks that provide a step-by-step walkthrough of:

- **Exploratory Data Analysis (EDA):**  
  Visualize soil and crop characteristics using boxplots, line charts, and heatmaps. Understand feature distributions, correlations, and outliers.

- **Feature Engineering:**  
  Creation of derived columns such as soil texture class, soil fertility level, and periodic encodings (sine/cosine) for time-based features.

- **Model Building:**  
  Training and tuning of classification models (e.g., Random Forest, XGBoost) for soil categorization, and regression models for yield prediction.

- **Hyperparameter Tuning:**  
  Optimization of model parameters using `GridSearchCV` and `RandomizedSearchCV`.

- **Model Evaluation:**  
  Detailed metrics for each model (e.g., R², RMSE, MAE) and cross-validation results to ensure robustness.

> These notebooks are ideal for exploring the data science workflow in detail and reproducing the results step by step.


---

## Future Enhancements
- Add real-time weather and irrigation integration.
- Incorporate satellite image features.
- Deploy interactive dashboards.

---

## Contributors
- **Madhav Betha**
- **Sumanth Gannamani**
- **Khyathi Narra**
