# Leak-Free Classification Pipeline

## Overview
This project provides a clean, beginner-friendly demonstration of constructing an end-to-end classification machine learning pipeline using scikit-learn on the Titanic passenger survival dataset. The project highlights the critical distinction between correct, leak-free pipeline validation and standard pitfalls that lead to data contamination.

## Objective
The objective is to build a binary classification model where all preprocessing steps (missing value imputation, feature scaling, and one-hot encoding) and the classifier are wrapped into **one single scikit-learn Pipeline**. The project demonstrates proper validation using 5-fold cross-validation and proves how isolating the test set prevents data leakage from inflating model evaluation.

## Dataset
- **Name**: Titanic Dataset
- **Source**: Reliable public repository (Datascience Dojo / GitHub & Kaggle standard distribution)
- **Target Variable**: `Survived` (`0` = Did not survive, `1` = Survived)
- **Total Samples**: 891 rows (80% Train: 712 samples, 20% Test: 179 samples)
- **Removed Columns**: `PassengerId`, `Name`, `Ticket`, `Cabin` (dropped due to lack of generalizable predictive power or excessive missingness)

## Features
- **Numerical Features**:
  - `Age`: Passenger age in years (contains missing values)
  - `Fare`: Passenger ticket fare
  - `SibSp`: Number of siblings / spouses aboard the Titanic
  - `Parch`: Number of parents / children aboard the Titanic
- **Categorical Features**:
  - `Sex`: Passenger sex (`male`, `female`)
  - `Embarked`: Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton, contains missing values)
  - `Pclass`: Ticket class (`1` = 1st, `2` = 2nd, `3` = 3rd)

## Model
- **Algorithm**: `LogisticRegression(max_iter=1000, random_state=42)`

## Preprocessing
Preprocessing is implemented via a modular `ColumnTransformer` combining two separate sub-pipelines:
- **Numerical**:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- **Categorical**:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

## Cross-Validation
- **Strategy**: 5-fold `KFold(n_splits=5, shuffle=True, random_state=42)`
- Evaluated on `X_train` and `y_train` using `cross_val_score` with `scoring="accuracy"`

## Data Leakage Prevention
Preprocessing is wrapped inside the `Pipeline`. Consequently:
1. Preprocessing transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) calculate statistics (median, mean, standard deviation, and categorical levels) **only from the training split of each cross-validation fold**.
2. Validation folds and the final test set are transformed using parameters learned strictly from training data, never contributing to the calculation of those parameters.
3. This completely prevents information leakage from the validation folds or the held-out test set into model training.

## Final Evaluation
Below are the **actual results** obtained from running `classification_pipeline_cv.ipynb`:

### 5-Fold Cross-Validation Scores
| Fold | Accuracy Score | Accuracy (%) |
| :--- | :---: | :---: |
| Fold 1 | 0.7832 | 78.32% |
| Fold 2 | 0.7552 | 75.52% |
| Fold 3 | 0.7887 | 78.87% |
| Fold 4 | 0.8028 | 80.28% |
| Fold 5 | 0.8732 | 87.32% |
| **Mean CV Accuracy** | **0.8007** | **80.07%** |
| **CV Standard Deviation** | **0.0394** | **3.94%** |

### Test Set Performance (Held-Out Test Set: 179 samples)
| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Final Test Accuracy** | **0.8045** | **80.45%** |
| **Final Test Precision** | **0.7931** | **79.31%** |
| **Final Test Recall** | **0.6667** | **66.67%** |
| **Final Test F1-Score** | **0.7244** | **72.44%** |

### Confusion Matrix (Test Set)
```text
                     Predicted: 0   Predicted: 1
Actual: 0 (Died)          98             12
Actual: 1 (Survived)      23             46
```

## How to Run

```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
# Option A: Run via Python module (works reliably across all systems):
python -m notebook classification_pipeline_cv.ipynb

# Or using the classic jupyter launcher:
python -m jupyter notebook classification_pipeline_cv.ipynb

# Option B: Run directly (if your Python Scripts folder is in PATH):
jupyter notebook classification_pipeline_cv.ipynb
```

> **Tip (VS Code / Antigravity IDE)**: You can also simply click on `classification_pipeline_cv.ipynb` in the file explorer to open, view, and run cells directly within your IDE.
