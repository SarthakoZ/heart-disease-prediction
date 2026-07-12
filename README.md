# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease from
patient clinical data, comparing **K-Nearest Neighbors** and **Logistic
Regression** classifiers.

## Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, and
early risk identification from routine clinical measurements (blood
pressure, cholesterol, BMI, etc.) can support earlier intervention. This
project frames heart disease detection as a **binary classification**
problem: given a patient's clinical profile, predict whether they are
likely to have heart disease (`1`) or not (`0`).

> ⚠️ **Disclaimer:** This project uses a small, custom/synthetic dataset for
> learning purposes. It is **not** a validated clinical tool and should not
> be used for real medical diagnosis or decision-making.

## Dataset

`data/Heart_Disease_Custom_Dataset_600_Rows.xlsx` — 600 patient records with
the following columns:

| Column            | Description                              |
|-------------------|-------------------------------------------|
| Patient_ID        | Unique patient identifier                 |
| Age               | Age in years                              |
| Sex               | M / F                                     |
| Chest_Pain        | Type of chest pain                        |
| Resting_BP        | Resting blood pressure                    |
| Cholesterol       | Serum cholesterol                         |
| Fasting_BS        | Fasting blood sugar (0/1)                 |
| Max_Heart_Rate    | Maximum heart rate achieved               |
| Exercise_Angina   | Exercise-induced angina (Yes/No)          |
| BMI               | Body mass index                           |
| Smoking           | Smoking status (Yes/No)                   |
| Family_History    | Family history of heart disease (Yes/No)  |
| Heart_Disease     | Target label (0 = No, 1 = Yes)            |

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── Heart_Disease_Custom_Dataset_600_Rows.xlsx
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb   # exploratory / walkthrough notebook
├── src/
│   ├── train.py                         # trains models, saves artifacts
│   └── predict.py                       # loads artifacts, runs inference
├── models/
│   ├── model.pkl                        # final trained Logistic Regression model
│   ├── scaler.pkl                       # fitted MinMaxScaler
│   ├── encoder.pkl                      # fitted OneHotEncoder (chest_pain)
│   └── label_encoders.pkl               # fitted LabelEncoders (sex, smoking, etc.)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/SarthakoZ/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

## Usage

**Train the models from scratch** (reproduces the notebook results and
overwrites the saved artifacts in `models/`):

```bash
python src/train.py
```

**Run a prediction** on a new patient record using the saved model:

```bash
python src/predict.py
```

Edit the `sample_patient` dictionary in `src/predict.py`, or import the
`predict()` function directly in your own code:

```python
from src.predict import predict

patient = {
    "age": 54, "sex": "m", "chest_pain": "typical",
    "resting_bp": 130, "cholesterol": 246, "fasting_bs": 0,
    "max_heart_rate": 150, "exercise_angina": "no",
    "bmi": 27.5, "smoking": "no", "family_history": "yes",
}
print(predict(patient))  # "0" or "1"
```

## Pipeline

1. **Cleaning** — drop duplicates, lowercase column names and categorical
   values, treat negative numeric values as missing.
2. **Encoding** — label-encode binary categorical columns (`sex`,
   `exercise_angina`, `smoking`, `family_history`); one-hot encode
   `chest_pain`.
3. **Scaling** — `MinMaxScaler` applied to all features.
4. **Modeling** — train/test split (80/20, `random_state=42`), then fit
   both a `KNeighborsClassifier` (k=5) and a `LogisticRegression` model.
5. **Evaluation** — accuracy, precision, recall, and confusion matrix for
   each model.

## Results

| Model               | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| KNN (k=5)            | 84.2%    | 92.3%     | 87.5%  |
| Logistic Regression  | **96.7%**| 97.9%     | 97.9%  |

**Logistic Regression** was selected as the final model based on its higher
accuracy, precision, and recall.

## Reproducibility

- Train/test split uses `random_state=42`, so re-running `src/train.py`
  should reproduce the same split and near-identical metrics.
- The saved artifacts in `models/` (`model.pkl`, `scaler.pkl`,
  `encoder.pkl`, `label_encoders.pkl`) must always be used **together** —
  the scaler and encoders were fit on the exact same training split as the
  model, and `predict.py` depends on that consistency.
- If you modify the feature set or preprocessing steps, re-run
  `src/train.py` to regenerate all four artifacts before using
  `src/predict.py`.

## Limitations & Future Work

- Dataset is relatively small (600 rows) and may not generalize to broader
  populations.
- No hyperparameter tuning was performed (e.g. `GridSearchCV` for `k` in
  KNN, or regularization strength in Logistic Regression).
- No cross-validation — results are based on a single train/test split.
- Possible next steps: try tree-based models (Random Forest, XGBoost),
  add cross-validation, and evaluate with ROC-AUC in addition to accuracy/
  precision/recall.

## Author

**SarthakoZ** — [github.com/SarthakoZ](https://github.com/SarthakoZ)
