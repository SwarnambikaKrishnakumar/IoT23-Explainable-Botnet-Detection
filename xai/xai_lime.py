import os
import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# IoT BOTNET DETECTION
# COMPLETE EXPLAINABLE AI MODULE
#
# XAI METHODS:
# 1. LIME - Local explanation
# 2. Surrogate Decision Tree - Global model interpretation
# 3. Permutation sensitivity - Feature importance
# ============================================================


print()
print("=" * 70)
print(" COMPLETE EXPLAINABLE AI MODULE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print()
print("[1/9] Loading processed data...")

X_train = pd.read_csv(
    "processed_data/X_train.csv"
)

X_test = pd.read_csv(
    "processed_data/X_test.csv"
)

y_train = pd.read_csv(
    "processed_data/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "processed_data/y_test.csv"
).squeeze()


def convert_label(value):

    value = str(value).strip().lower()

    if value == "benign":
        return 0

    return 1


y_train_binary = y_train.apply(
    convert_label
)

y_test_binary = y_test.apply(
    convert_label
)


feature_names = X_train.columns.tolist()


print("Training data :", X_train.shape)
print("Testing data  :", X_test.shape)
print("Features      :", len(feature_names))


# ============================================================
# 2. TRAIN LOF
# ============================================================

print()
print("[2/9] Training final LOF model...")


X_train_benign = X_train[
    y_train_binary == 0
]


lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.03,
    novelty=True
)


lof.fit(
    X_train_benign
)


print(
    "Benign training samples:",
    len(X_train_benign)
)


# ============================================================
# 3. MODEL PREDICTIONS
# ============================================================

print()
print("[3/9] Generating LOF predictions...")


test_predictions_raw = lof.predict(
    X_test
)


# LOF:
# +1 = inlier / benign
# -1 = outlier / malicious

test_predictions = np.where(
    test_predictions_raw == -1,
    1,
    0
)


accuracy = accuracy_score(
    y_test_binary,
    test_predictions
)

precision = precision_score(
    y_test_binary,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_binary,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_binary,
    test_predictions,
    zero_division=0
)


cm = confusion_matrix(
    y_test_binary,
    test_predictions
)


tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)


print()
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")
print("FPR      :", round(fpr * 100, 2), "%")


# ============================================================
# 4. SELECT TEST SAMPLE FOR LIME
# ============================================================

print()
print("[4/9] Selecting sample for local explanation...")


malicious_indices = np.where(
    y_test_binary.values == 1
)[0]


if len(malicious_indices) == 0:

    sample_index = 0

else:

    sample_index = int(
        malicious_indices[0]
    )


sample = X_test.iloc[
    sample_index
].values


actual_label = int(
    y_test_binary.iloc[
        sample_index
    ]
)


predicted_label = int(
    test_predictions[
        sample_index
    ]
)


actual_text = (
    "Malicious"
    if actual_label == 1
    else "Benign"
)


predicted_text = (
    "Malicious"
    if predicted_label == 1
    else "Benign"
)


print(
    "Selected sample:",
    sample_index
)

print(
    "Actual:",
    actual_text
)

print(
    "Prediction:",
    predicted_text
)


# ============================================================
# 5. LIME
# ============================================================

print()
print("[5/9] Generating LIME explanation...")


def lime_predict(data):

    data = pd.DataFrame(
        data,
        columns=feature_names
    )

    predictions = lof.predict(
        data
    )

    scores = lof.decision_function(
        data
    )

    # Convert LOF anomaly score into
    # a smooth explanation probability.
    malicious_probability = (
        1 /
        (
            1 +
            np.exp(scores)
        )
    )

    benign_probability = (
        1 -
        malicious_probability
    )

    return np.column_stack(
        [
            benign_probability,
            malicious_probability
        ]
    )


explainer = LimeTabularExplainer(

    training_data=
        X_train_benign.values,

    feature_names=
        feature_names,

    class_names=[
        "Benign",
        "Malicious"
    ],

    mode="classification",

    discretize_continuous=True,

    random_state=42
)


lime_explanation = (
    explainer.explain_instance(
        sample,
        lime_predict,
        num_features=10
    )
)


lime_features = (
    lime_explanation.as_list()
)


print(
    "LIME explanation generated."
)


# ============================================================
# 6. SURROGATE DECISION TREE
# ============================================================

print()
print("[6/9] Training surrogate Decision Tree...")


# Generate LOF predictions for training data.
#
# These predictions are used as the target for
# the surrogate model.

train_lof_raw = lof.predict(
    X_train
)


train_lof_predictions = np.where(
    train_lof_raw == -1,
    1,
    0
)


surrogate = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


surrogate.fit(
    X_train,
    train_lof_predictions
)


surrogate_predictions = surrogate.predict(
    X_test
)


surrogate_agreement = accuracy_score(
    test_predictions,
    surrogate_predictions
)


tree_rules = export_text(
    surrogate,
    feature_names=feature_names
)


print(
    "Surrogate agreement:",
    round(
        surrogate_agreement * 100,
        2
    ),
    "%"
)


# ============================================================
# 7. PERMUTATION SENSITIVITY
# ============================================================

print()
print("[7/9] Calculating global feature importance...")


# To keep execution fast, use a representative
# subset of the test data.

importance_sample_size = min(
    500,
    len(X_test)
)


importance_indices = np.arange(
    importance_sample_size
)


X_importance = X_test.iloc[
    importance_indices
].copy()


baseline_predictions = test_predictions[
    importance_indices
]


importance_results = []


rng = np.random.RandomState(
    42
)


for feature in feature_names:

    original_values = (
        X_importance[
            feature
        ].copy()
    )


    shuffled_values = (
        original_values
        .values.copy()
    )


    rng.shuffle(
        shuffled_values
    )


    X_permuted = (
        X_importance.copy()
    )


    X_permuted[
        feature
    ] = shuffled_values


    permuted_raw = (
        lof.predict(
            X_permuted
        )
    )


    permuted_predictions = (
        np.where(
            permuted_raw == -1,
            1,
            0
        )
    )


    baseline_agreement = (
        np.mean(
            baseline_predictions ==
            baseline_predictions
        )
    )


    permuted_agreement = (
        np.mean(
            baseline_predictions ==
            permuted_predictions
        )
    )


    importance = (
        baseline_agreement -
        permuted_agreement
    )


    importance_results.append(
        {
            "feature": feature,
            "importance": importance
        }
    )


importance_df = pd.DataFrame(
    importance_results
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "Global feature importance calculated."
)


# ============================================================
# 8. CREATE DATA FOR HTML
# ============================================================

print()
print("[8/9] Building professional XAI dashboard...")


# ------------------------------------------------------------
# LIME HTML
# ------------------------------------------------------------

lime_html = ""


max_lime_weight = max(
    abs(weight)
    for _, weight in lime_features
)


for feature, weight in lime_features:

    width = (
        abs(weight) /
        max_lime_weight *
        100
        if max_lime_weight != 0
        else 0
    )


    if weight >= 0:

        bar_class = "orange-bar"
        text_class = "orange-text"
        interpretation = (
            "Supports malicious prediction"
        )

    else:

        bar_class = "blue-bar"
        text_class = "blue-text"
        interpretation = (
            "Supports benign prediction"
        )


    lime_html += f"""

    <div class="lime-row">

        <div class="lime-header">

            <span class="feature-label">
                {feature}
            </span>

            <span class="weight {text_class}">
                {weight:+.5f}
            </span>

        </div>

        <div class="progress-track">

            <div
                class="{bar_class}"
                style="width:{width:.2f}%">
            </div>

        </div>

        <div class="contribution">
            {interpretation}
        </div>

    </div>

    """


# ------------------------------------------------------------
# GLOBAL IMPORTANCE HTML
# ------------------------------------------------------------

global_html = ""


top_global = (
    importance_df
    .head(10)
)


if len(top_global) > 0:

    max_importance = max(
        abs(
            top_global[
                "importance"
            ]
        )
    )

else:

    max_importance = 1


for _, row in top_global.iterrows():

    importance = row[
        "importance"
    ]


    width = (
        abs(importance) /
        max_importance *
        100
        if max_importance != 0
        else 0
    )


    global_html += f"""

    <div class="lime-row">

        <div class="lime-header">

            <span class="feature-label">
                {row["feature"]}
            </span>

            <span class="weight orange-text">
                {importance:.5f}
            </span>

        </div>

        <div class="progress-track">

            <div
                class="orange-bar"
                style="width:{width:.2f}%">
            </div>

        </div>

        <div class="contribution">
            Higher value indicates greater influence
            on LOF prediction stability.
        </div>

    </div>

    """


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

confusion_html = f"""

<div class="matrix">

    <div class="matrix-title">
        Confusion Matrix
    </div>

    <div class="matrix-grid">

        <div></div>

        <div class="matrix-axis">
            Predicted Benign
        </div>

        <div class="matrix-axis">
            Predicted Malicious
        </div>


        <div class="matrix-axis vertical">
            Actual Benign
        </div>

        <div class="matrix-cell tn">
            <strong>{tn}</strong>
            <span>True Negative</span>
        </div>

        <div class="matrix-cell fp">
            <strong>{fp}</strong>
            <span>False Positive</span>
        </div>


        <div class="matrix-axis vertical">
            Actual Malicious
        </div>

        <div class="matrix-cell fn">
            <strong>{fn}</strong>
            <span>False Negative</span>
        </div>

        <div class="matrix-cell tp">
            <strong>{tp}</strong>
            <span>True Positive</span>
        </div>

    </div>

</div>

"""


# ------------------------------------------------------------
# SURROGATE RULES
# ------------------------------------------------------------

tree_rules_html = (
    tree_rules
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)


# ============================================================
# 9. PROFESSIONAL HTML DASHBOARD
# ============================================================

html = f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">


<title>
IoT Botnet Detection | Explainable AI Dashboard
</title>


<style>

/* =========================================================
   GLOBAL
   ========================================================= */

* {{
    box-sizing: border-box;
}}

body {{

    margin: 0;

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;

    background:
        #f4f7fb;

    color:
        #172033;

}}


/* =========================================================
   HEADER
   ========================================================= */

.header {{

    background:
        linear-gradient(
            135deg,
            #102f52,
            #174b7a
        );

    color:
        white;

    padding:
        30px 55px;

    box-shadow:
        0 4px 20px
        rgba(0,0,0,.12);

}}

.header-inner {{

    max-width:
        1280px;

    margin:
        auto;

}}

.header h1 {{

    margin:
        0;

    font-size:
        29px;

    font-weight:
        650;

}}

.header p {{

    margin:
        8px 0 0;

    color:
        #dbeafe;

    font-size:
        14px;

}}


/* =========================================================
   NAVIGATION
   ========================================================= */

.nav {{

    background:
        white;

    border-bottom:
        1px solid #dce5ef;

}}

.nav-inner {{

    max-width:
        1280px;

    margin:
        auto;

    padding:
        13px 25px;

    display:
        flex;

    gap:
        25px;

}}

.nav span {{

    font-size:
        12px;

    font-weight:
        600;

    color:
        #64748b;

}}


/* =========================================================
   MAIN
   ========================================================= */

.container {{

    max-width:
        1280px;

    margin:
        32px auto;

    padding:
        0 25px;

}}


.section-heading {{

    margin:
        30px 0 16px;

    font-size:
        21px;

    font-weight:
        650;

    color:
        #123b63;

}}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpis {{

    display:
        grid;

    grid-template-columns:
        repeat(5,1fr);

    gap:
        15px;

}}

.kpi {{

    background:
        white;

    border:
        1px solid #dce5ef;

    border-radius:
        11px;

    padding:
        20px;

    box-shadow:
        0 3px 12px
        rgba(15,39,71,.06);

}}

.kpi-label {{

    font-size:
        11px;

    color:
        #718096;

    text-transform:
        uppercase;

    letter-spacing:
        .7px;

}}

.kpi-value {{

    margin-top:
        9px;

    font-size:
        22px;

    font-weight:
        700;

    color:
        #123b63;

}}

.kpi-orange {{

    color:
        #c65d00;

}}


/* =========================================================
   PANELS
   ========================================================= */

.panel {{

    background:
        white;

    border:
        1px solid #dce5ef;

    border-radius:
        11px;

    padding:
        27px;

    margin-top:
        22px;

    box-shadow:
        0 3px 12px
        rgba(15,39,71,.06);

}}

.panel h2 {{

    margin:
        0;

    color:
        #123b63;

    font-size:
        20px;

}}

.panel-description {{

    margin:
        8px 0 24px;

    color:
        #718096;

    font-size:
        13px;

    line-height:
        1.6;

}}


/* =========================================================
   TWO COLUMN
   ========================================================= */

.two-column {{

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        22px;

}}


/* =========================================================
   LIME
   ========================================================= */

.lime-row {{

    padding:
        13px 0;

    border-bottom:
        1px solid #edf1f5;

}}

.lime-row:last-child {{

    border-bottom:
        none;

}}

.lime-header {{

    display:
        flex;

    justify-content:
        space-between;

    gap:
        20px;

}}

.feature-label {{

    font-size:
        13px;

    font-weight:
        600;

    color:
        #334155;

}}

.weight {{

    font-size:
        13px;

    font-weight:
        700;

}}

.orange-text {{

    color:
        #c65d00;

}}

.blue-text {{

    color:
        #2563eb;

}}

.progress-track {{

    margin-top:
        7px;

    width:
        100%;

    height:
        9px;

    border-radius:
        7px;

    background:
        #e9eef4;

    overflow:
        hidden;

}}

.orange-bar {{

    height:
        100%;

    background:
        #d97706;

    border-radius:
        7px;

}}

.blue-bar {{

    height:
        100%;

    background:
        #2563eb;

    border-radius:
        7px;

}}

.contribution {{

    margin-top:
        5px;

    font-size:
        10px;

    color:
        #94a3b8;

}}


/* =========================================================
   RESULT BANNER
   ========================================================= */

.result-banner {{

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    background:
        #fff7ed;

    border:
        1px solid #fed7aa;

    border-radius:
        10px;

    padding:
        18px 20px;

}}

.result-label {{

    font-size:
        12px;

    color:
        #9a5b19;

}}

.result-value {{

    margin-top:
        4px;

    font-size:
        22px;

    font-weight:
        750;

    color:
        #c2410c;

}}


/* =========================================================
   CONFUSION MATRIX
   ========================================================= */

.matrix {{

    margin-top:
        20px;

}}

.matrix-title {{

    font-size:
        14px;

    font-weight:
        650;

    color:
        #334155;

    margin-bottom:
        15px;

}}

.matrix-grid {{

    display:
        grid;

    grid-template-columns:
        120px 1fr 1fr;

    gap:
        5px;

    max-width:
        600px;

}}

.matrix-axis {{

    padding:
        12px;

    text-align:
        center;

    font-size:
        11px;

    color:
        #64748b;

    font-weight:
        600;

}}

.matrix-axis.vertical {{

    display:
        flex;

    align-items:
        center;

    justify-content:
        flex-end;

}}

.matrix-cell {{

    min-height:
        90px;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        center;

    align-items:
        center;

    border-radius:
        7px;

}}

.matrix-cell strong {{

    font-size:
        24px;

}}

.matrix-cell span {{

    margin-top:
        4px;

    font-size:
        10px;

}}

.tn {{

    background:
        #eaf6ee;

    color:
        #20733d;

}}

.tp {{

    background:
        #e7f0fb;

    color:
        #205a93;

}}

.fp {{

    background:
        #fff4df;

    color:
        #9a6500;

}}

.fn {{

    background:
        #fdecec;

    color:
        #a52c2c;

}}


/* =========================================================
   TREE
   ========================================================= */

.tree-box {{

    background:
        #101b2d;

    color:
        #dbeafe;

    padding:
        22px;

    border-radius:
        9px;

    overflow:
        auto;

}}

.tree-box pre {{

    margin:
        0;

    font-family:
        Consolas,
        monospace;

    font-size:
        12px;

    line-height:
        1.6;

}}


/* =========================================================
   INFO
   ========================================================= */

.info-grid {{

    display:
        grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:
        15px;

}}

.info-card {{

    background:
        #f8fafc;

    border:
        1px solid #e2e8f0;

    border-radius:
        9px;

    padding:
        20px;

}}

.info-card h3 {{

    margin:
        0 0 8px;

    color:
        #123b63;

    font-size:
        15px;

}}

.info-card p {{

    margin:
        0;

    color:
        #64748b;

    font-size:
        12px;

    line-height:
        1.65;

}}


/* =========================================================
   TABLE
   ========================================================= */

table {{

    width:
        100%;

    border-collapse:
        collapse;

}}

th {{

    background:
        #eef3f8;

    color:
        #475569;

    padding:
        12px;

    text-align:
        left;

    font-size:
        11px;

}}

td {{

    padding:
        11px 12px;

    border-bottom:
        1px solid #edf1f5;

    font-size:
        12px;

}}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {{

    text-align:
        center;

    padding:
        30px;

    color:
        #94a3b8;

    font-size:
        11px;

}}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width:950px) {{

    .kpis {{
        grid-template-columns:
            repeat(2,1fr);
    }}

    .two-column {{
        grid-template-columns:
            1fr;
    }}

    .info-grid {{
        grid-template-columns:
            1fr;
    }}

}}

</style>

</head>


<body>


<!-- =====================================================
     HEADER
     ===================================================== -->

<div class="header">

    <div class="header-inner">

        <h1>
            IoT Botnet Detection — Explainable AI
        </h1>

        <p>
            Early-stage network anomaly detection with
            Local Outlier Factor and model explainability
        </p>

    </div>

</div>


<!-- =====================================================
     NAV
     ===================================================== -->

<div class="nav">

    <div class="nav-inner">

        <span>OVERVIEW</span>
        <span>MODEL PERFORMANCE</span>
        <span>LOCAL EXPLANATION</span>
        <span>GLOBAL IMPORTANCE</span>
        <span>SURROGATE MODEL</span>
        <span>INTERPRETATION</span>

    </div>

</div>


<!-- =====================================================
     MAIN
     ===================================================== -->

<div class="container">


<div class="section-heading">
    Detection Overview
</div>


<div class="kpis">


    <div class="kpi">

        <div class="kpi-label">
            Model
        </div>

        <div class="kpi-value">
            LOF
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-label">
            Accuracy
        </div>

        <div class="kpi-value">
            {accuracy*100:.2f}%
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-label">
            Precision
        </div>

        <div class="kpi-value">
            {precision*100:.2f}%
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-label">
            Recall
        </div>

        <div class="kpi-value">
            {recall*100:.2f}%
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-label">
            F1 Score
        </div>

        <div class="kpi-value">
            {f1*100:.2f}%
        </div>

    </div>


</div>


<!-- =====================================================
     DETECTION RESULT
     ===================================================== -->

<div class="panel">


    <div class="result-banner">

        <div>

            <div class="result-label">
                Selected Network Traffic Sample
            </div>

            <div class="result-value">
                {predicted_text.upper()}
            </div>

        </div>


        <div style="text-align:right">

            <div class="result-label">
                Actual Class
            </div>

            <div class="result-value">
                {actual_text.upper()}
            </div>

        </div>

    </div>


</div>


<!-- =====================================================
     MODEL PERFORMANCE
     ===================================================== -->

<div class="panel">

    <h2>
        Model Performance
    </h2>

    <div class="panel-description">
        Final evaluation of the Local Outlier Factor model
        on the held-out test dataset.
    </div>


    <div class="two-column">


        <div>

            <table>

                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>

                <tr>
                    <td>Accuracy</td>
                    <td>{accuracy*100:.2f}%</td>
                </tr>

                <tr>
                    <td>Precision</td>
                    <td>{precision*100:.2f}%</td>
                </tr>

                <tr>
                    <td>Recall</td>
                    <td>{recall*100:.2f}%</td>
                </tr>

                <tr>
                    <td>F1 Score</td>
                    <td>{f1*100:.2f}%</td>
                </tr>

                <tr>
                    <td>False Positive Rate</td>
                    <td>{fpr*100:.2f}%</td>
                </tr>

            </table>

        </div>


        <div>

            {confusion_html}

        </div>


    </div>

</div>


<!-- =====================================================
     LIME
     ===================================================== -->

<div class="panel">

    <h2>
        Local Explanation — LIME
    </h2>

    <div class="panel-description">

        LIME explains the selected individual prediction by
        estimating which input features locally support or
        oppose the model decision.

    </div>


    {lime_html}

</div>


<!-- =====================================================
     GLOBAL FEATURE IMPORTANCE
     ===================================================== -->

<div class="panel">

    <h2>
        Global Feature Importance
    </h2>

    <div class="panel-description">

        Permutation sensitivity measures how much the LOF
        prediction behavior changes when individual features
        are randomly shuffled.

    </div>


    {global_html}

</div>


<!-- =====================================================
     SURROGATE TREE
     ===================================================== -->

<div class="panel">

    <h2>
        Global Model Interpretation — Surrogate Decision Tree
    </h2>

    <div class="panel-description">

        A shallow Decision Tree is trained to approximate the
        predictions produced by the LOF model. This provides
        human-readable decision rules for understanding the
        model's overall behavior.

    </div>


    <div class="result-banner">

        <div>

            <div class="result-label">
                Surrogate Agreement with LOF
            </div>

            <div class="result-value">
                {surrogate_agreement*100:.2f}%
            </div>

        </div>


        <div>

            <div class="result-label">
                Tree Depth
            </div>

            <div class="result-value">
                4
            </div>

        </div>

    </div>


    <br>


    <div class="tree-box">

        <pre>
{tree_rules_html}
        </pre>

    </div>

</div>


<!-- =====================================================
     XAI METHODOLOGY
     ===================================================== -->

<div class="panel">

    <h2>
        Explainable AI Methodology
    </h2>

    <div class="panel-description">

        The explainability layer combines local and global
        interpretation techniques to improve transparency
        of the anomaly detection system.

    </div>


    <div class="info-grid">


        <div class="info-card">

            <h3>
                01 — LIME
            </h3>

            <p>
                Explains an individual network-flow prediction
                by identifying the local feature contributions
                that support or oppose the classification.
            </p>

        </div>


        <div class="info-card">

            <h3>
                02 — Permutation Sensitivity
            </h3>

            <p>
                Measures the influence of individual features
                by observing changes in LOF prediction behavior
                after feature values are randomly permuted.
            </p>

        </div>


        <div class="info-card">

            <h3>
                03 — Surrogate Model
            </h3>

            <p>
                Uses a shallow Decision Tree to approximate
                LOF decisions and produce human-readable
                decision rules.
            </p>

        </div>


    </div>

</div>


<!-- =====================================================
     INTERPRETATION
     ===================================================== -->

<div class="panel">

    <h2>
        XAI Interpretation
    </h2>

    <div class="panel-description">

        Key observations from the explainability analysis.
    </div>


    <div class="info-grid">


        <div class="info-card">

            <h3>
                Prediction
            </h3>

            <p>
                The selected network-traffic sample is classified
                as <strong>{predicted_text.lower()}</strong> by
                the Local Outlier Factor model.
            </p>

        </div>


        <div class="info-card">

            <h3>
                Local Explanation
            </h3>

            <p>
                LIME identifies the network-history and traffic
                features that contribute most strongly to the
                selected prediction.
            </p>

        </div>


        <div class="info-card">

            <h3>
                Model Transparency
            </h3>

            <p>
                The combination of local feature contributions,
                global feature sensitivity and surrogate rules
                provides multiple views of model behavior.
            </p>

        </div>


    </div>

</div>


<!-- =====================================================
     MODEL CONFIGURATION
     ===================================================== -->

<div class="panel">

    <h2>
        Detection & Explainability Configuration
    </h2>

    <div class="panel-description">
        Configuration used for the final explainability analysis.
    </div>


    <table>

        <tr>
            <th>Component</th>
            <th>Configuration</th>
        </tr>

        <tr>
            <td>Detection Algorithm</td>
            <td>Local Outlier Factor</td>
        </tr>

        <tr>
            <td>Number of Neighbors</td>
            <td>20</td>
        </tr>

        <tr>
            <td>Contamination</td>
            <td>0.03</td>
        </tr>

        <tr>
            <td>Novelty Detection</td>
            <td>Enabled</td>
        </tr>

        <tr>
            <td>LIME Features</td>
            <td>10</td>
        </tr>

        <tr>
            <td>Surrogate Tree Depth</td>
            <td>4</td>
        </tr>

        <tr>
            <td>Training Feature Count</td>
            <td>{len(feature_names)}</td>
        </tr>

        <tr>
            <td>Test Samples</td>
            <td>{len(X_test)}</td>
        </tr>

    </table>

</div>


</div>


<div class="footer">

    DEVELOPMENT OF AN EXPLAINABLE EARLY-STAGE IoT BOTNET DETECTION SYSTEM
    <br>
    Explainable AI Module • Local + Global Interpretation

</div>


</body>

</html>

"""


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "xai",
    exist_ok=True
)


output_file = (
    "xai/complete_xai_dashboard.html"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(html)


print()
print("=" * 70)
print(" COMPLETE XAI DASHBOARD CREATED")
print("=" * 70)

print()
print(
    "File:",
    output_file
)

print()
print(
    "XAI methods included:"
)

print(
    "1. LIME local explanation"
)

print(
    "2. Permutation feature sensitivity"
)

print(
    "3. Surrogate Decision Tree"
)

print()
print("Dashboard generation completed.")