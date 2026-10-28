"""
================================================================================
Flask Application: Medical Insurance Cost Category Prediction
Stage 3: Backend API and Prediction Functionality
================================================================================
"""

import os
import sys
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Import SQLite database helper
import database

# ==============================================================================
# FLASK APPLICATION CONFIGURATION
# ==============================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "college-medical-insurance-project-key-2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_pipeline.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Global model and metadata containers
MODEL_BUNDLE = None
METADATA = None


def load_model_and_metadata():
    """
    Loads the serialized Scikit-learn pipeline and metadata at server startup.
    """
    global MODEL_BUNDLE, METADATA

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Serialized model pipeline not found at: {MODEL_PATH}")
        print("Please run 'python train_model.py' first to train and serialize the model.")
        sys.exit(1)

    try:
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
        print(f"[SUCCESS] Loaded Model Pipeline: {MODEL_BUNDLE.get('model_name', 'Unknown')}")
    except Exception as e:
        print(f"[ERROR] Failed to load model pipeline: {e}")
        sys.exit(1)

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                METADATA = json.load(f)
            print("[SUCCESS] Loaded Model Metadata.")
        except Exception as e:
            print(f"[WARNING] Could not read metadata file: {e}")
            METADATA = {}
    else:
        METADATA = {}


# Execute model loading
load_model_and_metadata()


# ==============================================================================
# HELPER FUNCTIONS: VALIDATION & CLINICAL INTERPRETATION
# ==============================================================================

def validate_prediction_input(data):
    """
    Validates user input fields for prediction requests.
    Returns (cleaned_dict, error_message).
    """
    try:
        # 1. Validate Age
        if "age" not in data or data["age"] is None or str(data["age"]).strip() == "":
            return None, "Age is required."
        age = int(data["age"])
        if not (1 <= age <= 120):
            return None, "Age must be between 1 and 120 years."

        # 2. Validate Sex
        sex = str(data.get("sex", "")).strip().lower()
        if sex not in ["female", "male"]:
            return None, "Sex must be either 'female' or 'male'."

        # 3. Validate BMI
        if "bmi" not in data or data["bmi"] is None or str(data["bmi"]).strip() == "":
            return None, "BMI is required."
        bmi = float(data["bmi"])
        if not (10.0 <= bmi <= 70.0):
            return None, "BMI must be a realistic value between 10.0 and 70.0."

        # 4. Validate Children
        if "children" not in data or data["children"] is None or str(data["children"]).strip() == "":
            return None, "Number of children is required."
        children = int(data["children"])
        if not (0 <= children <= 15):
            return None, "Children count must be between 0 and 15."

        # 5. Validate Smoker
        smoker = str(data.get("smoker", "")).strip().lower()
        if smoker not in ["yes", "no"]:
            return None, "Smoker status must be 'yes' or 'no'."

        # 6. Validate Region
        region = str(data.get("region", "")).strip().lower()
        valid_regions = ["northeast", "northwest", "southeast", "southwest"]
        if region not in valid_regions:
            return None, f"Region must be one of: {', '.join(valid_regions)}."

        cleaned = {
            "age": age,
            "sex": sex,
            "bmi": round(bmi, 2),
            "children": children,
            "smoker": smoker,
            "region": region
        }
        return cleaned, None

    except ValueError as e:
        return None, f"Invalid numerical format: {str(e)}"
    except Exception as e:
        return None, f"Input processing error: {str(e)}"


def get_clinical_insights(age, bmi, smoker, category):
    """
    Generates health-risk explanations and personalized recommendations.
    """
    # BMI clinical category
    if bmi < 18.5:
        bmi_status = "Underweight (< 18.5)"
    elif bmi < 25.0:
        bmi_status = "Normal weight (18.5 - 24.9)"
    elif bmi < 30.0:
        bmi_status = "Overweight (25.0 - 29.9)"
    else:
        bmi_status = "Obese (≥ 30.0)"

    risk_factors = []
    recommendations = []

    if smoker == "yes":
        risk_factors.append("Active tobacco smoking (Major premium risk driver)")
        recommendations.append("Enrolling in a smoking cessation program can reduce health risks and insurance costs.")
    else:
        risk_factors.append("Non-smoker (Favorable health indicator)")

    if bmi >= 30.0:
        risk_factors.append(f"High Body Mass Index ({bmi} - {bmi_status})")
        recommendations.append("Cardiovascular fitness and nutrition counseling are advised to lower health risk.")
    elif bmi < 18.5:
        risk_factors.append(f"Low Body Mass Index ({bmi} - {bmi_status})")

    if age >= 50:
        risk_factors.append(f"Senior age category ({age} years)")

    if category == "Low":
        summary_text = "Favorable demographic and healthy lifestyle profiles place you in the lowest premium bracket."
        color_theme = "success"
    elif category == "Medium":
        summary_text = "Moderate insurance premium category, influenced by age or intermediate health indicators."
        color_theme = "warning"
    else:
        summary_text = "High premium category, driven primarily by lifestyle factors (e.g. smoking) or elevated BMI."
        color_theme = "danger"

    return {
        "bmi_status": bmi_status,
        "risk_factors": risk_factors,
        "recommendations": recommendations if recommendations else ["Maintain your current healthy lifestyle and balanced diet!"],
        "summary_text": summary_text,
        "color_theme": color_theme
    }


# ==============================================================================
# FLASK WEB ROUTES
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Renders the homepage.
    """
    return render_template(
        "index.html",
        metadata=METADATA,
        model_name=MODEL_BUNDLE.get("model_name", "Decision Tree") if MODEL_BUNDLE else "Model"
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction API endpoint:
    - Accepts JSON payload (AJAX/Fetch) or Form Data.
    - Validates inputs.
    - Evaluates ML pipeline.
    - Persists record to SQLite history.
    - Returns structured prediction response.
    """
    is_json = request.is_json
    raw_data = request.get_json(silent=True) if is_json else request.form.to_dict()

    if not raw_data:
        return jsonify({
            "status": "error",
            "message": "No input data provided."
        }), 400

    # 1. Validate inputs
    cleaned_data, error_msg = validate_prediction_input(raw_data)
    if error_msg:
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400

    # 2. Prepare DataFrame for model pipeline
    input_df = pd.DataFrame([cleaned_data])

    try:
        pipeline = MODEL_BUNDLE["pipeline"]
        classes = list(pipeline.classes_)

        # Generate prediction and probabilities
        predicted_category = str(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]

        prob_dict = {
            cls_name: round(float(prob) * 100, 2)
            for cls_name, prob in zip(classes, probabilities)
        }
        confidence = prob_dict.get(predicted_category, 0.0)

        # 3. Resolve Cost Dollar Range from Metadata
        thresholds = MODEL_BUNDLE.get("thresholds", {})
        low_bound = thresholds.get("low_cutoff", 6250.00)
        high_bound = thresholds.get("high_cutoff", 12733.36)

        if predicted_category == "Low":
            charges_range = f"<= ${low_bound:,.2f}"
        elif predicted_category == "Medium":
            charges_range = f"${low_bound:,.2f} - ${high_bound:,.2f}"
        else:
            charges_range = f"> ${high_bound:,.2f}"

        # 4. Clinical insights & health feedback
        insights = get_clinical_insights(
            cleaned_data["age"],
            cleaned_data["bmi"],
            cleaned_data["smoker"],
            predicted_category
        )

        # 5. Save prediction to SQLite database
        try:
            record_id = database.save_prediction(
                age=cleaned_data["age"],
                sex=cleaned_data["sex"],
                bmi=cleaned_data["bmi"],
                children=cleaned_data["children"],
                smoker=cleaned_data["smoker"],
                region=cleaned_data["region"],
                category=predicted_category,
                confidence=confidence,
                charges_range=charges_range
            )
        except Exception as db_err:
            print(f"[WARNING] Could not save prediction to SQLite: {db_err}")
            record_id = None

        response_payload = {
            "status": "success",
            "prediction": {
                "record_id": record_id,
                "category": predicted_category,
                "confidence": confidence,
                "charges_range": charges_range,
                "probabilities": prob_dict,
                "inputs": cleaned_data,
                "insights": insights
            }
        }

        # Return JSON if requested via AJAX or API
        if is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(response_payload), 200

        # Otherwise render template with results (for standard form POST fallback)
        return render_template(
            "index.html",
            metadata=METADATA,
            model_name=MODEL_BUNDLE.get("model_name", "Decision Tree"),
            result=response_payload["prediction"]
        )

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Inference error: {str(e)}"
        }), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Returns recent prediction history logs from SQLite database.
    """
    try:
        limit = request.args.get("limit", default=10, type=int)
        records = database.get_recent_predictions(limit=min(limit, 50))
        return jsonify({
            "status": "success",
            "count": len(records),
            "data": records
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """
    Returns analytics summary counts from SQLite database.
    """
    try:
        summary = database.get_analytics_dashboard_data()
        return jsonify({
            "status": "success",
            "data": summary
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """
    Returns comprehensive metrics, cross-tabulations, and trends for Chart.js dashboard.
    """
    try:
        analytics = database.get_analytics_dashboard_data()
        return jsonify({
            "status": "success",
            "data": analytics
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    """
    Resets the prediction history table to clean demo records.
    """
    try:
        database.clear_prediction_history()
        return jsonify({
            "status": "success",
            "message": "Prediction history successfully reset to clean demo records."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """
    Returns model training metadata, features, and cost categories.
    """
    return jsonify({
        "status": "success",
        "data": METADATA
    }), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Simple health verification endpoint.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_BUNDLE is not None,
        "model_name": MODEL_BUNDLE.get("model_name") if MODEL_BUNDLE else None
    }), 200


# ==============================================================================
# SERVER RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" MEDICAL INSURANCE PREDICTION FLASK APP RUNNING")
    print(" Access the application at: http://127.0.0.1:5000")
    print("=" * 75 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
