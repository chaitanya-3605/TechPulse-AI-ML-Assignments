# Assignment 9: Model Interpretability & Feature Importance Visualization

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Explainable AI (XAI), Random Forest MDI & Permutation Importance

---

## 1. Project Overview & Objective
This assignment explores machine learning interpretability by extracting, visualizing, and comparing **Mean Decrease in Impurity (MDI)** feature importances against out-of-sample **Permutation Feature Importance** for an automotive resale pricing Random Forest model.

---

## 2. Dataset & Problem Setting
- **Dataset:** 600 vehicle records across 7 features (`age_years`, `mileage_km`, `engine_size_L`, `horsepower`, `fuel_efficiency_kmpl`, `vehicle_weight_kg`, and one-hot encoded `brand`).
- **Model:** `RandomForestRegressor(n_estimators=300, random_state=42)`.

---

## 3. Interpretability Techniques Compared
1. **Tree-Based MDI (Gini Importance):**
   - Measures the total reduction of node impurity brought by each feature across all decision trees.
   - Fast to compute, but can exhibit bias toward high-cardinality features.
2. **Permutation Importance:**
   - Evaluates on the unseen test partition by randomly shuffling values of a single feature and recording the resulting decrease in model score ($R^2$).
   - Reflects true out-of-sample dependency without training impurity bias.

---

## 4. Key Analytical Findings
- **Primary Pricing Drivers:** `mileage_km` (~40.6%) and `age_years` (~38.8%) drive almost **80% of model predictive decisions**.
- **Secondary Drivers:** Engine output (`horsepower`) and luxury brand designation (`brand_BMW`).
- **Engineering Caveat:** Feature importance establishes statistical dependency within the model; it does not demonstrate underlying causal mechanics.

---

## 5. Execution Instructions
```bash
cd Assignments/Assignment_09
jupyter notebook TTL_Assignment_09.ipynb
```
