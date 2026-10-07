# Assignment 3: Data Cleaning & Preprocessing Pipeline

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Tabular Preprocessing, Outlier Handling & Feature Transformation

---

## 1. Project Overview & Objective
This assignment demonstrates an end-to-end data preprocessing and cleaning pipeline using `pandas` and `scikit-learn` to prepare a noisy, synthetic automotive dataset for downstream machine learning algorithms.

*Note: Focuses strictly on data transformation; machine learning model training is intentionally decoupled.*

---

## 2. Pipeline Stages
1. **Raw Dataset Profiling:** Inspection of data shapes, schema types, missing rates, and raw distributions.
2. **Missing Value Imputation:**
   - Continuous numerical features (`mileage_km`, `age_years`, `engine_size_L`, `price`): Imputed with **median** via `SimpleImputer` to prevent outlier skewing.
   - Categorical feature (`fuel_type`): Imputed with **most frequent** category.
3. **Outlier Detection & Capping (IQR Winsorization):**
   - Interquartile Range calculated: $IQR = Q_3 - Q_1$.
   - Bounds established at $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.
   - Extreme observations capped without dropping records to retain sample volume.
4. **Categorical Feature Encoding:** One-Hot Encoding applied to `fuel_type` (`drop='first'`) to prevent multicollinearity.
5. **Feature Standardization:** `StandardScaler` applied to all continuous attributes to normalize to zero mean and unit variance ($\mu=0, \sigma=1$).
6. **Model-Ready Dataset Assembly:** Concatenation into a final DataFrame (`processed_df`: 120 rows × 6 numerical columns) with zero missing entries.

---

## 3. Diagnostic Visualizations
- Comparative dual boxplots illustrating distributions **Before IQR Capping** vs. **After IQR Capping** across critical numerical fields.

---

## 4. Execution Instructions
```bash
cd Assignments/Assignment_03
jupyter notebook TTL_Assignment_03.ipynb
```
