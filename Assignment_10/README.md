# Assignment 10: Automotive MLOps Simulation — Lifecycle, Tracking & Deployment

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** MLOps, MLflow Experiment Tracking, Containerization & CI/CD

---

## 1. Project Overview & Objective
This assignment simulates an end-to-end educational **MLOps (Machine Learning Operations)** workflow for vehicle resale pricing, spanning model training, experiment logging with **MLflow**, artifact serialization (`rf_model.joblib`), standalone inference execution (`predict.py`), container packaging via **Docker**, and automated CI/CD via **GitHub Actions**.

---

## 2. MLOps Lifecycle Stages
```
[ Dataset Generation ] -> [ MLflow Experiment Tracking ] -> [ Export rf_model.joblib ]
                                                                      │
                      ┌───────────────────────────────────────────────┴────────────────────────┐
                      ▼                                                                        ▼
            [ predict.py Standalone ]                                                 [ Docker Container ]
                      │                                                                        │
                      └────────────────────────► [ GitHub Actions CI/CD ] ◄────────────────────┘
```

1. **Model Training & Experiment Tracking:**
   - Model: `RandomForestRegressor(n_estimators=200, max_depth=12)`.
   - Tracked in MLflow: Hyperparameters (`n_estimators`, `max_depth`), evaluation metrics (MAE, RMSE, $R^2$), and model artifact registry using local tracking backend (`sqlite:///mlflow.db`).
2. **Artifact Serialization:**
   - Exported model binary: `rf_model.joblib` (7.1 MB).
3. **Autonomous Inference Service (`predict.py`):**
   - Standalone inference script that loads `rf_model.joblib` and scores new vehicle feature payloads without requiring external MLflow server dependencies.
4. **Containerization (`Dockerfile`):**
   - Multi-layer lightweight Python 3.11-slim container bundling runtime dependencies, model weights, and inference code.
5. **Continuous Integration (`.github/workflows/mlops.yml`):**
   - Automated pipeline that installs requirements, executes headless notebook tests, and validates standalone inference outputs.

---

## 3. Local Verification Commands
1. **Run standalone inference:**
   ```bash
   cd Assignments/Assignment_10
   python predict.py
   ```
2. **Build and run Docker container:**
   ```bash
   docker build -t automotive-mlops:latest .
   docker run --rm automotive-mlops:latest
   ```

---

## 4. Execution Instructions
```bash
cd Assignments/Assignment_10
jupyter notebook TTL_Assignment_10.ipynb
```
