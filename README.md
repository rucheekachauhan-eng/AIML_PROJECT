# MediCost AI — Medical Insurance Cost Category Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://chartjs.org)

> **College Mini Project**: 3rd Year CSE (Artificial Intelligence & Machine Learning)  
> **Title**: Medical Insurance Cost Category Prediction Using Machine Learning  
> **Status**: **All 6 Stages Completed, Tested, and Verified**

---

## 1. Project Overview & Objective

Medical insurance premiums vary dramatically based on patient health status, lifestyle, and demographics. Continuous healthcare expenses follow a heavily right-skewed distribution driven by rare medical anomalies, making single-dollar regression volatile and unrepresentative.

Insurance underwriters solve this by categorizing policyholders into **actuarial risk bands** or **cost tiers**. The primary objective of this project is to build an end-to-end, production-ready machine learning system that:
1. Conducts thorough Exploratory Data Analysis (EDA) on the Medical Cost Personal Dataset.
2. Formulates three clinically balanced cost tiers (**Low**, **Medium**, **High**) strictly without data leakage.
3. Implements and compares machine learning classifiers (**Decision Tree** vs. **Random Forest**).
4. Packages a self-contained Scikit-Learn `ColumnTransformer` + classifier pipeline serialized with **Joblib**.
5. Deploys an interactive, healthcare-themed web application (Flask + HTML5/CSS3/Vanilla JS) featuring real-time **Chart.js** population telemetry and persistent **SQLite** audit logging.
6. Delivers automated unit testing and a comprehensive **College Viva Defense Guide**.

---

## 2. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER (FRONTEND)                         │
│  - HTML5 & Modern Vanilla CSS3 (Healthcare Theme)                           │
│  - Asynchronous Fetch API & Dynamic BMI Calculator                          │
│  - Real-Time Probability Visualizations & Chart.js 4.4 Telemetry Canvases   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP POST / GET (JSON)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER (FLASK API)                        │
│  - app.py: Request Routing, Input Validation & Error Handling               │
│  - Clinical Risk Insights Generator & Health Recommendations                │
│  - REST Endpoints: /predict, /api/analytics, /api/history, /api/health      │
└───────────────────────┬─────────────────────────────┬───────────────────────┘
                        │                             │
                        ▼                             ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│     MACHINE LEARNING PIPELINE        │  │      PERSISTENCE (SQLITE DB)      │
│  - models/best_model_pipeline.joblib │  │  - database/predictions.db        │
│  - ColumnTransformer:                │  │  - table: predictions             │
│    • StandardScaler (Age, BMI, Kids) │  │  - Live Audit Log & KPI Aggregates│
│    • OneHotEncoder (Sex, Smoke, Reg) │  └───────────────────────────────────┘
│  - Tuned DecisionTreeClassifier      │
│  - Accuracy: 90.67% | F1: 90.62%     │
└──────────────────────────────────────┘
```

---

## 3. Dataset Description

The project uses the standard **Medical Cost Personal Dataset** (`insurance.csv`).

- **Total Records**: 1,338 records (1,337 unique, 1 duplicate pruned)
- **Missing Values**: 0 (Clean, complete dataset)
- **Target Variable**: `charges` categorized into Low, Medium, High tiers

| Column Name | Data Type | Description | Values / Range |
|:---|:---|:---|:---|
| `age` | Integer | Age of primary beneficiary | 18 to 64 years |
| `sex` | Categorical | Biological sex of policyholder | `female`, `male` |
| `bmi` | Float | Body Mass Index ($\text{kg}/\text{m}^2$) | 15.96 to 53.13 |
| `children`| Integer | Dependent children covered on insurance | 0 to 5 |
| `smoker` | Categorical | Tobacco smoking status | `yes`, `no` |
| `region` | Categorical | US geographic residential zone | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges`| Float | Billed medical insurance costs (USD) | $1,121.87 to $63,770.43 |

---

## 4. Machine Learning Methodology

### A. Zero Data Leakage Stratification
To prevent test set information from contaminating the training process, the raw data was split **first** into **80% training** (1,069 samples) and **20% testing** (268 samples). Quantile boundaries were calculated **strictly from `charges_train`**:

| Cost Category | Quantile Percentile | Billed Charges Range (USD) | Train Samples (80%) | Test Samples (20%) |
|:---|:---|:---|:---:|:---:|
| **Low** | $0\% - 33.33\%$ | $\le \$6,250.00$ | 356 | 88 |
| **Medium**| $33.33\% - 66.67\%$ | $\$6,250.00 - \$12,733.36$ | 357 | 87 |
| **High** | $66.67\% - 100\%$ | $> \$12,733.36$ | 356 | 93 |

### B. Preprocessing (`ColumnTransformer`)
- **Numerical Features** (`age`, `bmi`, `children`): Standardized with `StandardScaler()` ($\mu = 0, \sigma = 1$).
- **Categorical Features** (`sex`, `smoker`, `region`): Encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')`.

### C. Model Comparison & Evaluation (Test Set Results)

Both models were regularized using hyperparameter constraints (`max_depth`, `min_samples_leaf`) to prevent overfitting:

| Metric | Decision Tree Classifier | Random Forest Classifier |
|:---|:---:|:---:|
| **Test Accuracy** | **90.67%** | 88.81% |
| **Macro Precision** | **91.06%** | 89.50% |
| **Macro Recall** | **90.85%** | 89.02% |
| **Macro F1-Score** | **90.62%** | 88.77% |
| **Weighted F1-Score** | **90.60%** | 88.74% |
| **Winner / Selection** | **Best Model (Selected)** | Runner-up |

#### Decision Tree Detailed Test Set Classification Report:
```text
              precision    recall  f1-score   support
         Low       0.90      0.98      0.93        88
      Medium       0.86      0.93      0.90        87
        High       0.97      0.82      0.89        93

    accuracy                           0.91       268
   macro avg       0.91      0.91      0.91       268
weighted avg       0.91      0.91      0.91       268
```

---

## 5. Project Directory Structure

```text
Medical-Insurance-Mini-Project/
├── dataset/
│   └── insurance.csv                   # Medical Cost Personal Dataset
├── database/
│   └── predictions.db                  # Persistent SQLite database
├── models/
│   ├── best_model_pipeline.joblib      # Serialized Scikit-Learn Pipeline
│   └── model_metadata.json             # Model metrics & schema metadata
├── notebooks/
│   └── eda_plots/                      # High-resolution saved visualizations
│       ├── 1_charges_distribution.png
│       ├── 2_charges_by_smoking_status.png
│       ├── 3_bmi_vs_charges_by_smoker.png
│       └── 4_confusion_matrices.png
├── static/
│   ├── css/
│   │   └── style.css                   # Healthcare CSS design system
│   └── js/
│       └── main.js                     # Fetch API controller, BMI calculator & Chart.js logic
├── templates/
│   └── index.html                      # Interactive web application template
├── app.py                              # Flask application backend
├── database.py                         # SQLite query & persistence layer
├── train_model.py                      # Complete ML training & evaluation script
├── test_project.py                     # Automated unit and integration test suite
├── requirements.txt                    # Project dependencies
├── VIVA_QUESTIONS_AND_ANSWERS.md       # Comprehensive viva defense & examiner guide
└── README.md                           # Project documentation
```

---

## 6. Installation & Execution Guide (VS Code)

### Step 1: Open Project in VS Code
Open VS Code, choose **File > Open Folder...**, and select the project directory. Open the terminal (**Terminal > New Terminal**).

### Step 2: Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate in Windows PowerShell
.\venv\Scripts\Activate.ps1

# (Alternative for Command Prompt)
# venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Pipeline (Stages 1 & 2)
```bash
python train_model.py
```
*Generates EDA charts in `notebooks/eda_plots/` and saves the model pipeline in `models/best_model_pipeline.joblib`.*

### Step 5: Run Automated Tests (Stage 6 QA)
```bash
python test_project.py
```
*Executes all 10 unit tests across dataset, ML pipeline, SQLite persistence, and Flask REST endpoints in under 0.3s.*

### Step 6: Start the Flask Web Server (Stages 3, 4, 5)
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 7. Web Application & API Documentation

### REST API Endpoints:

| Method | Endpoint | Description | Sample Output |
|:---|:---|:---|:---|
| `GET` | `/` | Serves main prediction application | HTML Document |
| `POST` | `/predict` | Evaluates patient metrics and returns predicted tier | JSON (`category`, `confidence`, `probabilities`, `insights`) |
| `GET` | `/api/analytics` | Returns aggregated telemetry for Chart.js | JSON (KPIs, Category Dist, Smoker Cross, BMI Dist, Timeline) |
| `GET` | `/api/history` | Returns recent prediction audit logs from SQLite | JSON (List of recent records) |
| `POST` | `/api/clear-history` | Resets SQLite database to clean demo records | JSON (`status: success`) |
| `GET` | `/api/metadata` | Returns model performance metrics and thresholds | JSON (`best_model`, `cost_thresholds`, `classes`) |
| `GET` | `/api/health` | Health verification endpoint | JSON (`status: healthy`, `model_loaded: true`) |

---

## 8. College Viva Preparation & Defense

For viva defense preparation, refer to the complete examiner Q&A guide:
👉 **[VIVA_QUESTIONS_AND_ANSWERS.md](file:///c:/Users/A/Desktop/AIML_PROJECT/VIVA_QUESTIONS_AND_ANSWERS.md)**

Includes:
- Elevator pitch and project problem statement
- Theoretical deep-dives (Gini Impurity, Information Gain, Macro F1 vs. Accuracy)
- Why classification was chosen over linear regression
- How data leakage was strictly avoided
- Why Decision Tree outperformed Random Forest
- 20+ Frequently Asked Viva Questions with model student answers
- Live viva demonstration checklist

---

## 9. Authors & Academic Credits

- **Student**: Third Year B.Tech CSE (Artificial Intelligence & Machine Learning)
- **Project Title**: Medical Insurance Cost Category Prediction Using Machine Learning
- **Institution**: College Mini Project Submission (2026)
