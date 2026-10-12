from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "processed_data")


X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")


# ============================================================
# LOAD DATA
# ============================================================

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]


# Convert labels safely
def convert_label(value):
    value = str(value).strip().lower()

    if value in ["0", "benign", "normal"]:
        return 0

    return 1


y_train = y_train.apply(convert_label).astype(int)
y_test = y_test.apply(convert_label).astype(int)


# Make sure test columns match training columns
X_test = X_test[X_train.columns]


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# LOF MODEL
# ============================================================

# Train only on benign traffic
benign_train_mask = y_train == 0

X_benign_train = X_train_scaled[benign_train_mask]


lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.03,
    novelty=True
)

lof.fit(X_benign_train)


# Predictions
y_pred = lof.predict(X_test_scaled)

# LOF gives:
# +1 = normal
# -1 = anomaly

y_pred_binary = np.where(y_pred == -1, 1, 0)


# Decision score
decision_scores = lof.decision_function(X_test_scaled)


# ============================================================
# LIME
# ============================================================

lime_explainer = LimeTabularExplainer(
    X_benign_train,
    feature_names=list(X_train.columns),
    class_names=["Benign", "Malicious"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


# ============================================================
# SAMPLE OPTIONS
# ============================================================

sample_options = []

# Use first 20 test samples.
# This guarantees that the dropdown always contains samples.

for idx in range(min(20, len(X_test))):

    actual = int(y_test.iloc[idx])
    predicted = int(y_pred_binary[idx])

    sample_options.append({
        "index": idx,
        "actual": "Malicious" if actual == 1 else "Benign",
        "predicted": "Malicious" if predicted == 1 else "Benign"
    })


# ============================================================
# HELPER
# ============================================================

def get_status(actual, predicted):

    if actual == predicted:
        return "Correct Prediction"

    if actual == 0 and predicted == 1:
        return "False Positive"

    if actual == 1 and predicted == 0:
        return "False Negative"

    return "Unknown"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    selected_sample = request.args.get("sample", default=0, type=int)

    if selected_sample < 0 or selected_sample >= len(X_test):
        selected_sample = 0

    # Actual feature values
    row = X_test.iloc[selected_sample]

    features = []

    for feature_name in X_test.columns:

        features.append({
            "name": feature_name,
            "value": row[feature_name]
        })

    actual = int(y_test.iloc[selected_sample])

    predicted = int(y_pred_binary[selected_sample])

    status = get_status(actual, predicted)

    return render_template(
        "index.html",
        sample_options=sample_options,
        selected_sample=selected_sample,
        features=features,
        actual_label="Malicious" if actual == 1 else "Benign",
        predicted_label="Malicious" if predicted == 1 else "Benign",
        status=status,
        decision_score=round(float(decision_scores[selected_sample]), 6)
    )


# ============================================================
# SELECT SAMPLE
# ============================================================

@app.route("/sample", methods=["POST"])
def select_sample():

    sample_number = request.form.get("sample_number", type=int)

    if sample_number is None:
        sample_number = 0

    if sample_number < 0 or sample_number >= len(X_test):
        sample_number = 0

    return redirect(
        url_for("home", sample=sample_number)
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    sample_number = request.form.get("sample_number", type=int)

    if sample_number is None:
        sample_number = 0

    if sample_number < 0 or sample_number >= len(X_test):
        sample_number = 0

    return redirect(
        url_for("home", sample=sample_number)
    )


# ============================================================
# XAI EXPLANATION
# ============================================================

@app.route("/explain", methods=["POST"])
def explain():

    sample_number = request.form.get("sample_number", type=int)

    if sample_number is None:
        sample_number = 0

    if sample_number < 0 or sample_number >= len(X_test):
        sample_number = 0

    # Exact scaled sample
    sample = X_test_scaled[sample_number]

    # Actual / predicted
    actual = int(y_test.iloc[sample_number])
    predicted = int(y_pred_binary[sample_number])

    actual_text = "Malicious" if actual == 1 else "Benign"
    predicted_text = "Malicious" if predicted == 1 else "Benign"

    status = get_status(actual, predicted)

    score = float(decision_scores[sample_number])

    # --------------------------------------------------------
    # LIME prediction function
    # --------------------------------------------------------

    def predict_for_lime(data):

        predictions = lof.predict(data)

        result = []

        for prediction in predictions:

            if prediction == -1:
                result.append([0.05, 0.95])

            else:
                result.append([0.95, 0.05])

        return np.array(result)

    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    explanation = lime_explainer.explain_instance(
        sample,
        predict_for_lime,
        num_features=10,
        top_labels=2
    )

    # Predicted class
    predicted_class = 1 if predicted == 1 else 0

    lime_list = explanation.as_list(
        label=predicted_class
    )

    # --------------------------------------------------------
    # Convert LIME output
    # --------------------------------------------------------

    lime_features = []

    for feature, contribution in lime_list:

        if contribution >= 0:
            direction = "Supports prediction"
        else:
            direction = "Opposes prediction"

        lime_features.append({
            "feature": feature,
            "contribution": round(float(contribution), 6),
            "direction": direction
        })

    # --------------------------------------------------------
    # Actual feature values
    # --------------------------------------------------------

    feature_values = []

    original_row = X_test.iloc[sample_number]

    for feature_name in X_test.columns:

        feature_values.append({
            "name": feature_name,
            "value": original_row[feature_name]
        })

    return render_template(
        "xai.html",

        sample_number=sample_number,

        actual_label=actual_text,

        predicted_label=predicted_text,

        status=status,

        decision_score=round(score, 6),

        lime_features=lime_features,

        feature_values=feature_values
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "running",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": len(X_train.columns),
        "model": "Local Outlier Factor",
        "n_neighbors": 20,
        "contamination": 0.03
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("EXPLAINABLE IoT BOTNET DETECTION SYSTEM")
    print("=" * 60)
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Features         : {len(X_train.columns)}")
    print("Model            : Local Outlier Factor")
    print("Neighbors        : 20")
    print("Contamination    : 0.03")
    print("=" * 60)
    print()
    print("Open this in your browser:")
    print("http://127.0.0.1:5000")
    print()

    port = int(os.environ.get("PORT", 5000))

app.run(
    host="0.0.0.0",
    port=port,
    debug=False
)