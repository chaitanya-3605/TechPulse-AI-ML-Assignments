# Assignment 4: Automotive Resale Price Prediction

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Regression Pipelines, ColumnTransformer & Ensemble Benchmarks

---

## 1. Project Overview & Objective
This assignment develops a modular machine learning regression workflow to predict automobile resale prices from mixed numerical and categorical features, leveraging scikit-learn `Pipeline` and `ColumnTransformer` to guarantee zero data leakage.

---

## 2. Dataset & Feature Schema
- **Volume:** Exactly 600 vehicle records.
- **Target Variable:** `price` (Vehicle resale price in USD).
- **Continuous Features:** `age_years`, `mileage_km`, `engine_size_L`.
- **Categorical Features:** `brand` (`Toyota`, `Honda`, `Ford`, `BMW`, `Hyundai`), `fuel_type` (`Petrol`, `Diesel`, `Electric`).

---

## 3. Modeling Architecture & Benchmark
A unified `ColumnTransformer` handles:
- `StandardScaler()` for continuous predictors.
- `OneHotEncoder(handle_unknown='ignore')` for categorical attributes.

Three candidate regression pipelines were evaluated on a hold-out test set (80/20 train/test split):

| Model Architecture | MAE (USD) | RMSE (USD) | $R^2$ Score | Rank |
|---|---|---|---|---|
| **Linear Regression** | **$1,770.83** | **$2,250.87** | **0.9505** | **1 (Best)** |
| Gradient Boosting Regressor | $1,972.54 | $2,446.63 | 0.9415 | 2 |
| Random Forest Regressor | $2,187.32 | $2,746.79 | 0.9263 | 3 |

**Outcome:** Ordinary Least Squares Linear Regression achieved the lowest RMSE ($2,250.87) and highest $R^2$ (0.9505), proving exceptional explanatory power on this physical pricing dataset.

---

## 4. Execution Instructions
```bash
cd Assignments/Assignment_04
jupyter notebook TTL_Assignment_04.ipynb
```
