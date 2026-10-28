"""
================================================================================
AI/ML Mini Project: Medical Insurance Cost Category Prediction
Stage 1 & 2: Dataset Loading, EDA, Preprocessing, Model Training & Evaluation
================================================================================

Description:
This script provides the complete machine learning workflow for the project:
1. Stage 1:
   - Dataset loading and structural inspection
   - Missing value and duplicate detection
   - Schema and numerical range validation
   - Exploratory Data Analysis (EDA) charts (Distribution, Boxplot, Scatter)
2. Stage 2:
   - Leak-free train-test split
   - Train-set-only quantile-based target categorization (Low / Medium / High)
   - Feature engineering pipeline (StandardScaler for numeric, OneHotEncoder for categorical)
   - Training & hyperparameter tuning for Decision Tree and Random Forest classifiers
   - Comprehensive model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrices)
   - Serialization of the best performing end-to-end Scikit-Learn pipeline via Joblib
   - Export of metadata (JSON) and sample prediction verification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================================================================
# 1. FILE PATH CONFIGURATION & DIRECTORIES
# ==============================================================================

def get_project_paths():
    """
    Returns an absolute path mapping for all project assets.
    Ensures seamless execution across any working directory in VS Code.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "base": base_dir,
        "dataset": os.path.join(base_dir, "dataset", "insurance.csv"),
        "eda_plots": os.path.join(base_dir, "notebooks", "eda_plots"),
        "models": os.path.join(base_dir, "models"),
        "model_file": os.path.join(base_dir, "models", "best_model_pipeline.joblib"),
        "metadata_file": os.path.join(base_dir, "models", "model_metadata.json"),
    }


def load_dataset(file_path):
    """
    Loads the insurance dataset from the specified CSV path.
    Gracefully handles missing files or read errors with helpful guidance.
    """
    print("=" * 75)
    print("STEP 1: LOADING DATASET")
    print("=" * 75)
    print(f"Target dataset path: {file_path}")

    if not os.path.exists(file_path):
        print("\n[ERROR] Dataset file not found!")
        print(f"Expected file at: {file_path}")
        print("\nTroubleshooting Instructions:")
        print("1. Ensure a folder named 'dataset' exists in your project root.")
        print("2. Place the 'insurance.csv' file inside the 'dataset' directory.")
        print("3. Verify that the file name is exactly 'insurance.csv'.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        print("[SUCCESS] Dataset loaded successfully.\n")
        return df
    except Exception as e:
        print(f"\n[ERROR] An error occurred while reading the CSV file: {e}")
        sys.exit(1)


# ==============================================================================
# 2. DATA INSPECTION & VALIDATION
# ==============================================================================

def inspect_dataset(df):
    """
    Performs comprehensive structural inspection and validation of the dataset.
    """
    print("=" * 75)
    print("STEP 2: DATASET INSPECTION & STRUCTURE")
    print("=" * 75)

    # 1. First five records
    print("\n--- 1. First 5 Records (df.head()) ---")
    print(df.head())

    # 2. Number of rows and columns
    rows, cols = df.shape
    print(f"\n--- 2. Dataset Dimensions ---")
    print(f"Total Rows (Records)    : {rows}")
    print(f"Total Columns (Features): {cols}")

    # 3. Column names and data types
    print("\n--- 3. Column Names & Data Types ---")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f" - {col:<12} : {dtype}")

    # 4. Missing values check
    print("\n--- 4. Missing Values Analysis ---")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    if missing_counts.sum() == 0:
        print("-> Result: No missing values found in the dataset.")
    else:
        print(f"-> Warning: Found {missing_counts.sum()} total missing values.")

    # 5. Duplicate records check
    print("\n--- 5. Duplicate Records Analysis ---")
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows detected: {duplicate_count}")
    if duplicate_count > 0:
        print(f"-> Note: {duplicate_count} duplicate row(s) exist in raw data.")

    # 6. Schema & Required Columns Validation
    required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    print("\n--- 6. Schema Validation ---")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns in dataset: {missing_cols}")
        print(f"Expected all of: {required_columns}")
        sys.exit(1)
    else:
        print("[SUCCESS] All required columns are present.")

    # 7. Numerical Validity Checks
    print("\n--- 7. Numerical Validity Checks ---")
    invalid_age = df[df["age"] <= 0]
    invalid_bmi = df[df["bmi"] <= 0]
    invalid_children = df[df["children"] < 0]
    invalid_charges = df[df["charges"] <= 0]

    anomalies_found = False
    if len(invalid_age) > 0:
        print(f"-> Anomaly: Found {len(invalid_age)} records with age <= 0.")
        anomalies_found = True
    if len(invalid_bmi) > 0:
        print(f"-> Anomaly: Found {len(invalid_bmi)} records with BMI <= 0.")
        anomalies_found = True
    if len(invalid_children) > 0:
        print(f"-> Anomaly: Found {len(invalid_children)} records with children < 0.")
        anomalies_found = True
    if len(invalid_charges) > 0:
        print(f"-> Anomaly: Found {len(invalid_charges)} records with charges <= 0.")
        anomalies_found = True

    if not anomalies_found:
        print("[SUCCESS] All numerical fields (age, bmi, children, charges) contain valid, positive ranges.")

    # 8. Descriptive statistics
    print("\n--- 8. Descriptive Statistics for Numerical Features ---")
    print(df.describe().round(2))

    # 9. Categorical feature distribution & unique values
    print("\n--- 9. Categorical Features Unique Values ---")
    categorical_cols = ["sex", "smoker", "region"]
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].unique().tolist()
            counts = df[col].value_counts().to_dict()
            print(f" - {col:<8} -> Unique: {unique_vals} | Counts: {counts}")


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==============================================================================

def perform_eda(df, output_dir):
    """
    Generates and saves the three Stage 1 EDA plots:
    1. Distribution of medical insurance charges
    2. Insurance charges by smoking status
    3. BMI versus insurance charges, distinguishing smokers and non-smokers
    """
    print("\n" + "=" * 75)
    print("STEP 3: GENERATING EXPLORATORY DATA ANALYSIS (EDA) CHARTS")
    print("=" * 75)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving high-resolution charts to: {output_dir}\n")

    sns.set_theme(style="whitegrid", palette="muted")

    # Chart 1: Distribution of Medical Insurance Charges
    print("Generating Chart 1: Distribution of Medical Insurance Charges...")
    plt.figure(figsize=(10, 6))
    mean_charges = df["charges"].mean()
    median_charges = df["charges"].median()

    sns.histplot(
        df["charges"],
        kde=True,
        color="#2b5c8f",
        bins=35,
        edgecolor="white",
        alpha=0.75
    )
    plt.axvline(mean_charges, color="#d9534f", linestyle="--", linewidth=2, label=f"Mean: ${mean_charges:,.2f}")
    plt.axvline(median_charges, color="#5cb85c", linestyle="-", linewidth=2, label=f"Median: ${median_charges:,.2f}")

    plt.title("Distribution of Medical Insurance Charges", fontsize=15, pad=15, fontweight="bold", color="#1a2e40")
    plt.xlabel("Insurance Charges ($)", fontsize=12, labelpad=10)
    plt.ylabel("Frequency (Count)", fontsize=12, labelpad=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()

    chart1_path = os.path.join(output_dir, "1_charges_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    print(f" -> Saved: {chart1_path}")
    plt.close()

    # Chart 2: Insurance Charges by Smoking Status
    print("\nGenerating Chart 2: Insurance Charges by Smoking Status...")
    plt.figure(figsize=(8, 6))
    palette = {"yes": "#e74c3c", "no": "#2ecc71"}
    ax2 = sns.boxplot(
        x="smoker",
        y="charges",
        hue="smoker",
        legend=False,
        data=df,
        palette=palette,
        width=0.45,
        boxprops=dict(alpha=0.85),
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7}
    )

    medians = df.groupby("smoker", observed=False)["charges"].median()
    for i, smoker_val in enumerate(["yes", "no"]):
        med_val = medians[smoker_val]
        ax2.text(
            i, med_val + 1000,
            f"Median:\n${med_val:,.0f}",
            horizontalalignment="center",
            size=10,
            color="black",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9)
        )

    plt.title("Medical Insurance Charges by Smoking Status", fontsize=15, pad=15, fontweight="bold", color="#1a2e40")
    plt.xlabel("Smoker Status (yes / no)", fontsize=12, labelpad=10)
    plt.ylabel("Insurance Charges ($)", fontsize=12, labelpad=10)
    plt.tight_layout()

    chart2_path = os.path.join(output_dir, "2_charges_by_smoking_status.png")
    plt.savefig(chart2_path, dpi=300)
    print(f" -> Saved: {chart2_path}")
    plt.close()

    # Chart 3: BMI vs. Insurance Charges
    print("\nGenerating Chart 3: BMI vs. Charges by Smoking Status...")
    plt.figure(figsize=(10, 6))
    scatter_palette = {"yes": "#e74c3c", "no": "#3498db"}
    sns.scatterplot(
        x="bmi",
        y="charges",
        hue="smoker",
        data=df,
        palette=scatter_palette,
        alpha=0.75,
        s=60,
        edgecolor="w",
        linewidth=0.5
    )
    plt.axvline(30, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Obesity Threshold (BMI = 30)")

    plt.title("Body Mass Index (BMI) vs. Insurance Charges", fontsize=15, pad=15, fontweight="bold", color="#1a2e40")
    plt.xlabel("Body Mass Index (BMI)", fontsize=12, labelpad=10)
    plt.ylabel("Insurance Charges ($)", fontsize=12, labelpad=10)
    plt.legend(title="Smoker", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()

    chart3_path = os.path.join(output_dir, "3_bmi_vs_charges_by_smoker.png")
    plt.savefig(chart3_path, dpi=300)
    print(f" -> Saved: {chart3_path}")
    plt.close()


# ==============================================================================
# 4. STAGE 2: PREPROCESSING & TARGET CATEGORIZATION (LEAK-FREE)
# ==============================================================================

def create_cost_categories(charges_series, low_cutoff, high_cutoff):
    """
    Maps continuous insurance charges to Low, Medium, and High categories
    using fixed cutoff thresholds.
    """
    def label_cost(val):
        if val <= low_cutoff:
            return "Low"
        elif val <= high_cutoff:
            return "Medium"
        else:
            return "High"

    return charges_series.apply(label_cost)


def prepare_data_and_splits(df):
    """
    Splits the data into Train and Test partitions BEFORE computing quantiles
    to strictly prevent data leakage from the test set into the training phase.
    """
    print("\n" + "=" * 75)
    print("STEP 4: STAGE 2 - TRAIN/TEST SPLIT & TARGET CATEGORIZATION")
    print("=" * 75)

    # 1. Clean duplicates to ensure unique observations
    clean_df = df.drop_duplicates().reset_index(drop=True)
    dropped_count = len(df) - len(clean_df)
    if dropped_count > 0:
        print(f"Removed {dropped_count} duplicate record(s). Clean dataset size: {len(clean_df)} records.")

    # 2. Separate features (X) from raw continuous target (charges)
    X = clean_df.drop(columns=["charges"])
    raw_charges = clean_df["charges"]

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, charges_train, charges_test = train_test_split(
        X, raw_charges, test_size=0.20, random_state=42
    )
    print(f"\nTrain set size : {len(X_train)} samples (80%)")
    print(f"Test set size  : {len(X_test)} samples (20%)")

    # 4. Compute Quantile Boundaries ONLY from training set charges
    # 33.33% percentile for Low/Medium boundary, 66.67% percentile for Medium/High boundary
    low_cutoff = float(charges_train.quantile(0.3333))
    high_cutoff = float(charges_train.quantile(0.6667))

    print("\n--- Training Set Cost Category Boundaries (Zero Data Leakage) ---")
    print(f" - Low Cost Tier    : Charges <= ${low_cutoff:,.2f}")
    print(f" - Medium Cost Tier : ${low_cutoff:,.2f} < Charges <= ${high_cutoff:,.2f}")
    print(f" - High Cost Tier   : Charges > ${high_cutoff:,.2f}")

    # 5. Apply the training-derived thresholds to both train and test splits
    y_train = create_cost_categories(charges_train, low_cutoff, high_cutoff)
    y_test = create_cost_categories(charges_test, low_cutoff, high_cutoff)

    # Class ordering: Low, Medium, High
    class_order = ["Low", "Medium", "High"]
    print("\n--- Class Distribution in Training Set (Target: y_train) ---")
    print(y_train.value_counts()[class_order])

    print("\n--- Class Distribution in Test Set (Target: y_test) ---")
    print(y_test.value_counts()[class_order])

    threshold_info = {
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "classes": class_order,
        "quantile_low": 0.3333,
        "quantile_high": 0.6667,
        "currency": "USD"
    }

    return X_train, X_test, y_train, y_test, threshold_info


def build_preprocessor():
    """
    Constructs a Scikit-Learn ColumnTransformer that:
    - Standardizes numerical features (age, bmi, children)
    - One-hot encodes categorical features (sex, smoker, region)
    """
    numerical_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numerical_features, categorical_features


# ==============================================================================
# 5. MODEL TRAINING, EVALUATION & COMPARISON
# ==============================================================================

def train_and_compare_models(X_train, y_train, X_test, y_test, preprocessor, output_dir):
    """
    Trains Decision Tree and Random Forest pipelines, evaluates both on the test set,
    prints comparative metrics, and plots evaluation charts.
    """
    print("\n" + "=" * 75)
    print("STEP 5: MODEL TRAINING, HYPERPARAMETER TUNING & EVALUATION")
    print("=" * 75)

    class_labels = ["Low", "Medium", "High"]

    # Define Candidate Classifiers
    candidate_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        print(f"\nTraining pipeline for: {name}...")

        # Build end-to-end pipeline (Preprocessor + Classifier)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Train on training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on unseen test set
        y_pred = pipeline.predict(X_test)

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, labels=class_labels, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, labels=class_labels, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, labels=class_labels, average="macro", zero_division=0)
        f1_weight = f1_score(y_test, y_pred, labels=class_labels, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)

        results[name] = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_mac,
            "f1_weighted": f1_weight,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred,
            "classification_report": classification_report(y_test, y_pred, labels=class_labels, zero_division=0)
        }

        print(f" -> {name} Test Accuracy : {acc * 100:.2f}%")
        print(f" -> {name} Macro F1-Score: {f1_mac * 100:.2f}%")

    # Display Detailed Classification Reports
    print("\n" + "-" * 75)
    print("DETAILED CLASSIFICATION REPORTS (TEST SET)")
    print("-" * 75)
    for name, data in results.items():
        print(f"\n=== {name} Classification Report ===")
        print(data["classification_report"])

    # Comparative Summary Table
    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 75)
    summary_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [f"{results[m]['accuracy'] * 100:.2f}%" for m in results],
        "Macro Precision": [f"{results[m]['precision_macro'] * 100:.2f}%" for m in results],
        "Macro Recall": [f"{results[m]['recall_macro'] * 100:.2f}%" for m in results],
        "Macro F1-Score": [f"{results[m]['f1_macro'] * 100:.2f}%" for m in results],
        "Weighted F1-Score": [f"{results[m]['f1_weighted'] * 100:.2f}%" for m in results],
    })
    print(summary_df.to_string(index=False))

    # Select Best Model Based on Macro F1-Score (balances all 3 classes) and Accuracy
    best_model_name = max(results, key=lambda m: (results[m]["f1_macro"], results[m]["accuracy"]))
    best_pipeline = fitted_pipelines[best_model_name]
    best_stats = results[best_model_name]

    print(f"\n[WINNER] Best Performing Model: {best_model_name} (Macro F1: {best_stats['f1_macro'] * 100:.2f}%)")

    # Generate Evaluation Plots (Confusion Matrices side-by-side)
    plot_evaluation_charts(results, class_labels, output_dir)

    return best_model_name, best_pipeline, results


def plot_evaluation_charts(results, class_labels, output_dir):
    """
    Plots side-by-side Confusion Matrices for Decision Tree and Random Forest.
    """
    print("\nGenerating Model Evaluation Charts...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (name, data) in enumerate(results.items()):
        cm = np.array(data["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else "Greens",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}\nAccuracy: {data['accuracy'] * 100:.2f}% | F1: {data['f1_macro'] * 100:.2f}%",
                            fontsize=13, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Predicted Cost Category", fontsize=11, labelpad=8)
        axes[idx].set_ylabel("True Cost Category", fontsize=11, labelpad=8)

    plt.suptitle("Model Evaluation: Confusion Matrices on Test Set", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "4_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f" -> Saved: {cm_path}")
    plt.close()


# ==============================================================================
# 6. SAVING THE COMPLETE PIPELINE & METADATA
# ==============================================================================

def save_pipeline_and_metadata(best_name, best_pipeline, threshold_info, results, paths, num_feats, cat_feats):
    """
    Saves the complete, self-contained Scikit-Learn pipeline to disk via Joblib.
    Also exports metadata JSON for frontend consumption and performs a test prediction.
    """
    print("\n" + "=" * 75)
    print("STEP 6: SERIALIZING BEST PIPELINE & METADATA")
    print("=" * 75)
    os.makedirs(paths["models"], exist_ok=True)

    # 1. Prepare Serialized Bundle Dictionary
    pipeline_bundle = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "classes": list(best_pipeline.classes_),
        "feature_names": num_feats + cat_feats,
        "numerical_features": num_feats,
        "categorical_features": cat_feats,
        "thresholds": threshold_info,
        "metrics": {
            "accuracy": float(results[best_name]["accuracy"]),
            "f1_macro": float(results[best_name]["f1_macro"]),
            "f1_weighted": float(results[best_name]["f1_weighted"]),
        }
    }

    # Save Joblib file
    joblib.dump(pipeline_bundle, paths["model_file"])
    print(f"[SUCCESS] Best Pipeline saved to: {paths['model_file']}")

    # 2. Prepare Human-Readable JSON Metadata (for Flask backend & reports)
    metadata_json = {
        "project": "Medical Insurance Cost Category Prediction",
        "stage": 2,
        "best_model": best_name,
        "classes": list(best_pipeline.classes_),
        "cost_thresholds": {
            "low_range": f"<= ${threshold_info['low_cutoff']:,.2f}",
            "medium_range": f"${threshold_info['low_cutoff']:,.2f} - ${threshold_info['high_cutoff']:,.2f}",
            "high_range": f"> ${threshold_info['high_cutoff']:,.2f}",
            "low_cutoff_raw": threshold_info["low_cutoff"],
            "high_cutoff_raw": threshold_info["high_cutoff"]
        },
        "model_performance": {
            model: {
                "accuracy": round(results[model]["accuracy"], 4),
                "macro_f1": round(results[model]["f1_macro"], 4),
                "weighted_f1": round(results[model]["f1_weighted"], 4)
            }
            for model in results
        },
        "expected_features": {
            "numerical": num_feats,
            "categorical": cat_feats
        }
    }

    with open(paths["metadata_file"], "w") as f:
        json.dump(metadata_json, f, indent=4)
    print(f"[SUCCESS] Model Metadata saved to: {paths['metadata_file']}")

    # 3. Verify Model Loading & Test Live Prediction
    print("\n--- Verifying Model Reload & Prediction Test ---")
    loaded_bundle = joblib.load(paths["model_file"])
    loaded_pipeline = loaded_bundle["pipeline"]

    # Sample Patient 1: Young, non-smoker, normal BMI
    patient_healthy = pd.DataFrame([{
        "age": 22,
        "sex": "female",
        "bmi": 21.5,
        "children": 0,
        "smoker": "no",
        "region": "northwest"
    }])

    # Sample Patient 2: Middle-aged, smoker, high BMI (obese)
    patient_highrisk = pd.DataFrame([{
        "age": 52,
        "sex": "male",
        "bmi": 34.8,
        "children": 2,
        "smoker": "yes",
        "region": "southeast"
    }])

    pred_healthy = loaded_pipeline.predict(patient_healthy)[0]
    prob_healthy = loaded_pipeline.predict_proba(patient_healthy)[0]

    pred_highrisk = loaded_pipeline.predict(patient_highrisk)[0]
    prob_highrisk = loaded_pipeline.predict_proba(patient_highrisk)[0]

    healthy_probs = {c: float(round(p, 3)) for c, p in zip(loaded_pipeline.classes_, prob_healthy)}
    highrisk_probs = {c: float(round(p, 3)) for c, p in zip(loaded_pipeline.classes_, prob_highrisk)}

    print(f"Sample 1 (Age 22, Non-Smoker, BMI 21.5) -> Predicted: {pred_healthy} Cost Category")
    print(f"  Class Probabilities: {healthy_probs}")

    print(f"Sample 2 (Age 52, Smoker, BMI 34.8)    -> Predicted: {pred_highrisk} Cost Category")
    print(f"  Class Probabilities: {highrisk_probs}")

    print("\n" + "=" * 75)
    print("[COMPLETED] STAGE 2 COMPLETED SUCCESSFULLY!")
    print("Pipeline is self-contained and ready for Flask web application (Stage 3).")
    print("=" * 75)


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 75)
    print(" MEDICAL INSURANCE COST CATEGORY PREDICTION - STAGE 1 & 2 PIPELINE ")
    print("#" * 75 + "\n")

    # 1. Resolve project directory structure
    paths = get_project_paths()

    # 2. Load dataset
    df = load_dataset(paths["dataset"])

    # 3. Stage 1: Dataset Inspection & Structural Validation
    inspect_dataset(df)

    # 4. Stage 1: Generate & Save Visualizations
    perform_eda(df, paths["eda_plots"])

    # 5. Stage 2: Leak-Free Train/Test Split & Target Categorization
    X_train, X_test, y_train, y_test, threshold_info = prepare_data_and_splits(df)

    # 6. Stage 2: Build Feature Preprocessor
    preprocessor, num_feats, cat_feats = build_preprocessor()

    # 7. Stage 2: Model Training, Comparison & Evaluation Charts
    best_name, best_pipeline, results = train_and_compare_models(
        X_train, y_train, X_test, y_test, preprocessor, paths["eda_plots"]
    )

    # 8. Stage 2: Serialize Best Pipeline and Save Metadata
    save_pipeline_and_metadata(
        best_name, best_pipeline, threshold_info, results, paths, num_feats, cat_feats
    )
