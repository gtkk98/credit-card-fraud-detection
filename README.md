# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised learning techniques. The project includes data preprocessing, handling class imbalance with SMOTE, model training, evaluation, and deployment through a FastAPI REST API.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small fraction of all transactions.

This project explores multiple machine learning approaches to identify fraudulent transactions while maximizing fraud detection performance through precision, recall, and F1-score metrics.

---

## 📊 Dataset

* **Dataset:** Credit Card Fraud Detection Dataset (Kaggle)
* **Characteristics:**
  * 284,807 transactions
  * 492 fraudulent transactions
  * 30 anonymized numerical features
  * Highly imbalanced target variable

### Target Class
* `0` = Legitimate Transaction
* `1` = Fraudulent Transaction

---

## 🛠️ Tech Stack

### Machine Learning
* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)

### Backend & API
* FastAPI
* Pydantic

### Testing & Serialization
* Pytest
* Joblib

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── api/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── random_forest.pkl
│
├── src/
│   └── credit_card_fraud_detection/
│       ├── preprocessing.py
│       ├── train.py
│       ├── smote.py
│       ├── train_random_forest.py
│       ├── tune_random_forest.py
│       ├── evaluate.py
│       └── model_io.py
│
├── tests/
├── notebooks/
│   └── 01_eda.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ✨ Features

* Exploratory Data Analysis (EDA)
* Data validation and preprocessing
* Feature scaling using `StandardScaler`
* Train/Test data splitting with stratification
* Class balancing using SMOTE
* Logistic Regression baseline model
* Random Forest classifier
* Model evaluation using:
  * Precision
  * Recall
  * F1 Score
  * Accuracy
* Model persistence using Joblib
* FastAPI REST API deployment
* Unit testing with Pytest

---

## 🔄 Machine Learning Pipeline

```text
Raw Dataset
    │
    ▼
Data Validation
    │
    ▼
Train/Test Split
    │
    ▼
Feature Scaling
    │
    ▼
SMOTE
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Saving
    │
    ▼
FastAPI Deployment
```

---

## 📈 Model Performance

### Logistic Regression (Baseline)
```text
Accuracy  : 0.9991
Precision : 0.8267
Recall    : 0.6327
F1 Score  : 0.7168
```

### Logistic Regression + SMOTE
```text
Accuracy  : 0.9741
Precision : 0.0578
Recall    : 0.9184
F1 Score  : 0.1088
```

### Random Forest (Best Model ✅)
```text
Accuracy  : 0.9995
Precision : 89.01% (0.8901)
Recall    : 82.65% (0.8265)
F1 Score  : 85.71% (0.8571)
```

---

## 🚀 Running the Project

### 1. Clone Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create Virtual Environment
Using `uv`:
```bash
uv venv
```

**Activate Environment:**
* **PowerShell:**
  ```powershell
  .venv\Scripts\activate
  ```
* **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Train the Model
```bash
python main.py
```
*This script will load data, preprocess, apply SMOTE, train models, evaluate performance, and save the trained model.*

### 5. Start the API
```bash
uvicorn api.main:app --reload
```

Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the interactive Swagger UI.

---

## 📩 Example API Request & Response

### Request (`POST /predict`)
```json
{
  "Time": 0,
  "V1": 0.1,
  "V2": 0.2,
  "V3": -0.1,
  "V4": 0.3,
  "V5": -0.2,
  "V6": 0.4,
  "V7": -0.1,
  "V8": 0.2,
  "V9": 0.1,
  "V10": -0.3,
  "V11": 0.4,
  "V12": 0.1,
  "V13": -0.2,
  "V14": 0.3,
  "V15": 0.1,
  "V16": -0.1,
  "V17": 0.2,
  "V18": 0.3,
  "V19": 0.1,
  "V20": 0.2,
  "V21": -0.1,
  "V22": 0.4,
  "V23": 0.2,
  "V24": -0.3,
  "V25": 0.1,
  "V26": 0.2,
  "V27": -0.1,
  "V28": 0.3,
  "Amount": 100.0
}
```

### Response
```json
{
  "fraud": false,
  "probability": 0.03
}
```

---

## 🧪 Testing

Run automated tests using Pytest:
```bash
pytest tests -v
```

---

## 💡 Key Learnings

* Handling highly imbalanced datasets.
* Applying SMOTE for oversampling.
* Comparing classification models.
* Evaluating fraud detection systems beyond simple accuracy.
* Building production-style ML pipelines.
* Deploying machine learning models with FastAPI.
* Writing automated tests for ML workflows.

---

## 🔮 Future Improvements

* [ ] Implement XGBoost classifier.
* [ ] Add Docker containerization support.
* [ ] Set up CI/CD pipeline with GitHub Actions.