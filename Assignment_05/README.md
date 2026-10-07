# Assignment 5: Predictive Maintenance from Telemetry Sensor Logs

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Binary Classification, Class Imbalance & Diagnostic Metrics

---

## 1. Project Overview & Objective
This assignment addresses industrial and automotive predictive maintenance: detecting imminent machine component failures from multi-channel sensor logs using a **Random Forest Classifier** tuned for class-imbalanced environments.

---

## 2. Dataset & Sensor Attributes
- **Volume:** Exactly 800 synthetic telemetry records.
- **Sensor Signals:** `temperature_C`, `vibration_mm_s`, `pressure_psi`, `rpm`, `run_hours`.
- **Target:** `failure` (Binary: `0 = Healthy`, `1 = Failure`).
- **Class Imbalance:** 74.4% Healthy (595 samples) vs. 25.6% Failure (205 samples).

---

## 3. Imbalance Mitigation & Validation Integrity
- **Stratified Partitioning:** 75/25 train/test split utilizing `stratify=y` to preserve rare class proportions in both sets.
- **Data Leakage Shield:** `StandardScaler` fit strictly on training telemetry data.
- **Algorithmic Weighting:** `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`.

---

## 4. Evaluation & Operational Interpretation
- **Overall Accuracy:** 81.0% on hold-out test set.
- **Healthy Class:** Precision = 0.82, Recall = 0.95, F1-score = 0.88.
- **Failure Class:** Precision = 0.73, Recall = 0.37, F1-score = 0.49.
- **Confusion Matrix Analysis:**
  - True Negatives (TN): 142
  - False Positives (FP): 7 (minor inspection overhead)
  - False Negatives (FN): 32 (critical missed breakdowns)
  - True Positives (TP): 19
- **Engineering Conclusion:** In critical maintenance systems, False Negatives carry catastrophic downtime costs compared to False Positives; thus sensitivity tuning and threshold optimization remain paramount.

---

## 5. Execution Instructions
```bash
cd Assignments/Assignment_05
jupyter notebook TTL_Assignment_05.ipynb
```
