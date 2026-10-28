"""
================================================================================
SQLite Database Helper for Medical Insurance Prediction History & Analytics
Stage 5: Analytics Dashboard & Telemetry
================================================================================
Manages persistent storage of user predictions and provides analytics queries
for the Chart.js visual dashboard.
Uses Python's standard library sqlite3.
"""

import os
import sqlite3
from datetime import datetime

# Resolve database path relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "predictions.db")


def get_db_connection():
    """
    Establishes and returns a SQLite connection with row factory enabled
    so query results can be accessed like dictionaries.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the predictions table if it does not already exist,
    and seeds initial demo records if the table is empty.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex TEXT NOT NULL,
                bmi REAL NOT NULL,
                children INTEGER NOT NULL,
                smoker TEXT NOT NULL,
                region TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                charges_range TEXT NOT NULL
            )
        """)
        conn.commit()

        # Seed initial realistic demonstration records if table is empty
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        if cursor.fetchone()["count"] == 0:
            seed_initial_demo_records(cursor)
            conn.commit()


def seed_initial_demo_records(cursor):
    """
    Populates initial realistic demo predictions so the dashboard displays
    meaningful healthcare visual charts out-of-the-box.
    """
    demo_samples = [
        ("2026-10-28 10:15:20", 22, "female", 21.5, 0, "no", "northwest", "Low", 94.0, "<= $6,250.00"),
        ("2026-10-28 10:45:10", 28, "male", 24.2, 1, "no", "southwest", "Low", 91.5, "<= $6,250.00"),
        ("2026-10-28 11:20:00", 35, "female", 26.8, 2, "no", "northeast", "Medium", 88.0, "$6,250.00 - $12,733.36"),
        ("2026-10-28 12:10:45", 46, "male", 29.4, 1, "no", "southeast", "Medium", 89.2, "$6,250.00 - $12,733.36"),
        ("2026-10-28 13:05:15", 52, "male", 34.8, 2, "yes", "southeast", "High", 100.0, "> $12,733.36"),
        ("2026-10-28 14:15:30", 38, "female", 32.1, 0, "yes", "northwest", "High", 98.4, "> $12,733.36"),
        ("2026-10-28 14:40:00", 24, "male", 23.0, 0, "no", "northeast", "Low", 95.0, "<= $6,250.00"),
        ("2026-10-28 15:00:25", 60, "female", 28.5, 3, "no", "southwest", "Medium", 86.5, "$6,250.00 - $12,733.36"),
        ("2026-10-28 15:30:10", 49, "male", 36.5, 1, "yes", "southeast", "High", 100.0, "> $12,733.36")
    ]
    cursor.executemany("""
        INSERT INTO predictions (
            timestamp, age, sex, bmi, children, smoker, region,
            predicted_category, confidence, charges_range
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, demo_samples)


def save_prediction(age, sex, bmi, children, smoker, region, category, confidence, charges_range):
    """
    Inserts a newly completed prediction into the SQLite database.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, age, sex, bmi, children, smoker, region,
                predicted_category, confidence, charges_range
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, int(age), sex, float(bmi), int(children), smoker, region,
            category, float(confidence), charges_range
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_predictions(limit=15):
    """
    Fetches the most recent predictions ordered by latest timestamp first.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, age, sex, bmi, children, smoker, region,
                   predicted_category, confidence, charges_range
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_analytics_dashboard_data():
    """
    Computes comprehensive aggregated metrics for the Chart.js Analytics Dashboard:
    1. Overall summary KPIs (total count, average confidence, average age, smoker count).
    2. Category distribution (Low, Medium, High).
    3. Smoking vs. Cost Category cross-tabulation.
    4. Regional volume breakdown.
    5. Age group distribution.
    6. BMI health classification breakdown.
    7. Recent confidence score timeline.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Total Count & Averages
        cursor.execute("""
            SELECT COUNT(*) as total,
                   AVG(confidence) as avg_conf,
                   AVG(age) as avg_age,
                   AVG(bmi) as avg_bmi
            FROM predictions
        """)
        overall = cursor.fetchone()
        total_count = overall["total"] if overall and overall["total"] else 0
        avg_confidence = round(float(overall["avg_conf"]), 1) if overall and overall["avg_conf"] else 0.0
        avg_age = round(float(overall["avg_age"]), 1) if overall and overall["avg_age"] else 0.0
        avg_bmi = round(float(overall["avg_bmi"]), 1) if overall and overall["avg_bmi"] else 0.0

        # 2. Cost Category Distribution
        cursor.execute("""
            SELECT predicted_category, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_category
        """)
        cat_rows = cursor.fetchall()
        cat_counts = {row["predicted_category"]: row["count"] for row in cat_rows}

        category_distribution = {
            "Low": cat_counts.get("Low", 0),
            "Medium": cat_counts.get("Medium", 0),
            "High": cat_counts.get("High", 0)
        }

        # 3. Smoker Breakdown by Category (Cross-tab)
        cursor.execute("""
            SELECT smoker, predicted_category, COUNT(*) as count
            FROM predictions
            GROUP BY smoker, predicted_category
        """)
        smoker_cat_rows = cursor.fetchall()
        smoker_cross = {
            "smoker": {"Low": 0, "Medium": 0, "High": 0},
            "non_smoker": {"Low": 0, "Medium": 0, "High": 0}
        }
        for r in smoker_cat_rows:
            group = "smoker" if r["smoker"] == "yes" else "non_smoker"
            cat = r["predicted_category"]
            if cat in smoker_cross[group]:
                smoker_cross[group][cat] = r["count"]

        # 4. Regional Distribution
        cursor.execute("""
            SELECT region, COUNT(*) as count
            FROM predictions
            GROUP BY region
        """)
        region_counts = {row["region"]: row["count"] for row in cursor.fetchall()}

        # 5. Age Group Distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN age < 30 THEN '18-29'
                    WHEN age <= 45 THEN '30-45'
                    WHEN age <= 60 THEN '46-60'
                    ELSE '60+'
                END AS age_group,
                COUNT(*) as count
            FROM predictions
            GROUP BY age_group
        """)
        age_counts = {row["age_group"]: row["count"] for row in cursor.fetchall()}
        age_distribution = {
            "18-29": age_counts.get("18-29", 0),
            "30-45": age_counts.get("30-45", 0),
            "46-60": age_counts.get("46-60", 0),
            "60+": age_counts.get("60+", 0)
        }

        # 6. BMI Category Distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN bmi < 25.0 THEN 'Normal (< 25)'
                    WHEN bmi < 30.0 THEN 'Overweight (25-29.9)'
                    ELSE 'Obese (>= 30)'
                END AS bmi_group,
                COUNT(*) as count
            FROM predictions
            GROUP BY bmi_group
        """)
        bmi_counts = {row["bmi_group"]: row["count"] for row in cursor.fetchall()}
        bmi_distribution = {
            "Normal (< 25)": bmi_counts.get("Normal (< 25)", 0),
            "Overweight (25-29.9)": bmi_counts.get("Overweight (25-29.9)", 0),
            "Obese (>= 30)": bmi_counts.get("Obese (>= 30)", 0)
        }

        # 7. Timeline of Last 8 Predictions
        cursor.execute("""
            SELECT id, timestamp, predicted_category, confidence
            FROM predictions
            ORDER BY id ASC
            LIMIT 10
        """)
        timeline_rows = cursor.fetchall()
        timeline = [
            {
                "id": r["id"],
                "time": r["timestamp"].split(" ")[1] if " " in r["timestamp"] else r["timestamp"],
                "category": r["predicted_category"],
                "confidence": round(float(r["confidence"]), 1)
            }
            for r in timeline_rows
        ]

        return {
            "kpis": {
                "total_predictions": total_count,
                "avg_confidence": avg_confidence,
                "avg_age": avg_age,
                "avg_bmi": avg_bmi,
                "smoker_count": sum(smoker_cross["smoker"].values()),
                "non_smoker_count": sum(smoker_cross["non_smoker"].values())
            },
            "category_distribution": category_distribution,
            "smoker_cross": smoker_cross,
            "region_distribution": region_counts,
            "age_distribution": age_distribution,
            "bmi_distribution": bmi_distribution,
            "timeline": timeline
        }


def clear_prediction_history():
    """
    Clears all recorded predictions and re-inserts clean demo seed records.
    Useful for resetting the dashboard during college demonstrations/viva.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        seed_initial_demo_records(cursor)
        conn.commit()


# Initialize database on module import
init_db()
