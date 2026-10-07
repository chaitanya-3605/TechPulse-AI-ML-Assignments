# Assignment 1: Machine Learning Model for Vehicle Fuel Efficiency (MPG) Estimation

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Supervised Machine Learning & Regression Analysis

---

## 1. Project Overview & Objective
The goal of this assignment is to design, train, and benchmark supervised regression models to estimate vehicle fuel efficiency in Miles Per Gallon (**MPG**) based on mechanical and technical specifications. Two distinct modeling approaches are evaluated and contrasted:
1. **Ordinary Least Squares (OLS) Linear Regression** with z-score feature normalization.
2. **Random Forest Regression** (ensemble of 100 decision trees) to capture potential non-linear relationships.

---

## 2. Dataset Description
- **Source:** Synthetic automotive engineering dataset containing 500 records.
- **File Location:** `data/car_mileage_dataset.csv`
- **Reproducibility:** Seeded with `np.random.seed(42)` for deterministic reproduction.
- **Physical Dynamics:** Modeled according to vehicle dynamics where curb weight and displacement inversely impact fuel economy, while newer model years incorporate efficiency gains.

### Feature Attributes:
| Feature | Type | Units / Range | Description |
|---|---|---|---|
| `cylinders` | Integer | 4, 6, 8 | Number of engine cylinders |
| `displacement`| Continuous | cu. inches | Total cylinder swept volume |
| `horsepower` | Continuous | Brake HP | Peak engine power output |
| `weight` | Continuous | lbs | Vehicle curb weight |
| `acceleration`| Continuous | seconds | Time required to accelerate from 0 to 60 mph |
| `model_year` | Integer | 70 to 82 | Model release year (1970–1982) |
| **`mpg`** (Target)| Continuous | Miles/Gallon | Fuel economy rating |

---

## 3. Preprocessing & Validation Setup
- **Train/Test Partition:** 80% training partition (400 samples) and 20% hold-out test partition (100 samples) using `random_state=42`.
- **Feature Standardization:** `StandardScaler` fitted strictly on training data (`X_train`) and transformed onto `X_test` to prevent data leakage.

---

## 4. Evaluation Metrics & Benchmark Results
Performance is evaluated across three primary regression metrics:
- **MAE (Mean Absolute Error):** Average magnitude of absolute residuals.
- **RMSE (Root Mean Squared Error):** Square root of mean squared error, sensitive to large deviations.
- **$R^2$ (Coefficient of Determination):** Proportion of variance explained by model predictors.

### Model Performance Comparison:
| Model Architecture | MAE (MPG) | RMSE (MPG) | $R^2$ Score | Best In |
|---|---|---|---|---|
| **Linear Regression** | **1.8651** | **2.3413** | **0.9143** | **All Metrics** |
| Random Forest Regressor | 1.9706 | 2.5538 | 0.8981 | Non-linear baseline |

**Key Takeaways:** Linear Regression slightly outperforms Random Forest on this dataset split due to the underlying physical linearity between vehicle mass/displacement and fuel consumption. Over 91.4% of the variance in fuel economy is successfully captured.

---

## 5. Artifacts and Generated Visualizations
All artifacts are automatically persisted in `outputs/`:
- `outputs/eda_mpg_distribution.png`: Histogram & KDE of fuel efficiency distribution.
- `outputs/eda_correlation_heatmap.png`: Pearson correlation matrix across attributes.
- `outputs/eda_features_vs_mpg.png`: Scatter relationships between technical specs and MPG.
- `outputs/model_comparison.csv`: Tabular benchmark results.
- `outputs/prediction_vs_actual.png`: Actual vs. predicted MPG scatter plots.
- `outputs/rf_feature_importance.png`: Feature importance breakdown from the Random Forest model.

---

## 6. Execution Instructions
1. Navigate to the assignment folder:
   ```bash
   cd Assignments/Assignment_01
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook TTL_Assignment_01.ipynb
   ```
