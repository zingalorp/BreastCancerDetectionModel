# Interpretable Breast Cancer Prediction

This repository demonstrates a pipeline for building a breast cancer detection model. The goal is to achieve high accuracy (F1 > 0.95) and provide visualization for error analysis and interpretability.

## Folder Structure

BREAST_CANCER_MODEL
├── data
│   ├── processed
│   │   ├── X_test_scaled.csv
│   │   ├── X_train_scaled.csv
│   │   ├── y_test.csv
│   │   └── y_train_res.csv
│   └── raw
│       └── Data for Task 1.csv
├── models
│   └── best_model.pkl
├── notebooks
│   ├── 1_eda.ipynb
│   ├── 2_modeling.ipynb
│   └── 3_interpretability_error_analysis.ipynb
├── reports
│   └── breast_cancer_report.html
│   └── figures
│       ├── confusion_matrix_error_analysis.png
│       ├── EDA_class_distribution.png
│       ├── EDA_feature_distributions.png
│       ├── EDA_full_correlation_heatmap.png
│       ├── logistic_coefficients.png
│       ├── shap_summary.png
│       └── shap_waterfall.png
├── src
│   ├── error_analysis.py
│   ├── interpretability.py
│   └── preprocessing.py
└── requirements.txt
└── README.txt


### `data/`
- **raw/** contains the original dataset (`Data for Task 1.csv`).
- **processed/** holds scaled and preprocessed train/test data files.

### `models/`
- Stores trained model objects.
- `best_model.pkl` is the top-performing model (Logistic Regression with an F1 of 0.98).

### `notebooks/`
- **1_eda.ipynb** covers exploratory data analysis.
- **2_modeling.ipynb** handles model selection, hyperparameter tuning, and performance comparison.
- **3_interpretability_error_analysis.ipynb** details model explainability (SHAP, coefficients) and error analysis.

### `reports/`
- **breast_cancer_report.html** presents an overview of the findings.
- **advanced_profiling/** may include supplementary profiling outputs.
- **figures/** contains all generated plots (e.g., SHAP plots, confusion matrix).

### `src/`
- **preprocessing.py** has functions for data cleaning, scaling, and balancing.
- **error_analysis.py** has functions for model error analysis and tables.
- **interpretability.py** has functions for interpretability plots.

### `requirements.txt`
Lists Python dependencies for consistent execution.

## Usage

1. Clone this repository.
2. Create a virtual environment.
3. Install dependencies with:

pip install -r requirements.txt

4. Run the notebooks in sequence:
1. `1_eda.ipynb`
2. `2_modeling.ipynb`
3. `3_interpretability_error_analysis.ipynb`

Each notebook saves relevant figures and outputs to the `reports/figures` directory.
