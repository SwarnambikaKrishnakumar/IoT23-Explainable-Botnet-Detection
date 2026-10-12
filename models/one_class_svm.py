import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

print("Loading preprocessed data...")

X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")

y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

print()
print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ============================================================
# 2. CONVERT LABELS TO BINARY FORMAT
# ============================================================

# 0 = Benign
# 1 = Malicious

def convert_label(label):
    label = str(label).strip().lower()

    if label == "benign":
        return 0
    else:
        return 1


y_train_binary = y_train.apply(convert_label)
y_test_binary = y_test.apply(convert_label)


print()
print("Training label distribution:")
print(
    pd.Series(y_train_binary)
    .map({0: "Benign", 1: "Malicious"})
    .value_counts()
)

print()
print("Testing label distribution:")
print(
    pd.Series(y_test_binary)
    .map({0: "Benign", 1: "Malicious"})
    .value_counts()
)


# ============================================================
# 3. CREATE VALIDATION SET
# ============================================================

# We keep the final test set completely untouched.
# Validation data is used only for selecting OSVM parameters.

X_train_model, X_validation, y_train_model, y_validation = train_test_split(
    X_train,
    y_train_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_train_binary
)

print()
print("Model-training data shape:", X_train_model.shape)
print("Validation data shape    :", X_validation.shape)


# ============================================================
# 4. USE ONLY BENIGN DATA FOR OSVM TRAINING
# ============================================================

X_train_benign = X_train_model[
    y_train_model == 0
]

print()
print("Benign samples used for OSVM training:", len(X_train_benign))


# ============================================================
# 5. HYPERPARAMETER SEARCH
# ============================================================

print()
print("Starting One-Class SVM hyperparameter search...")
print("Training is performed using BENIGN traffic only.")
print()

# Candidate gamma values
gamma_values = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5
]

# Candidate nu values
nu_values = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.10
]


best_score = -1
best_gamma = None
best_nu = None

search_results = []


for gamma in gamma_values:

    for nu in nu_values:

        print(
            f"Testing gamma={gamma}, nu={nu}"
        )

        try:

            model = OneClassSVM(
                kernel="rbf",
                gamma=gamma,
                nu=nu
            )

            # Train ONLY on benign traffic
            model.fit(X_train_benign)

            # Predict validation data
            validation_predictions = model.predict(
                X_validation
            )

            # OSVM:
            # +1 = normal / benign
            # -1 = anomaly / malicious

            y_validation_pred = np.where(
                validation_predictions == 1,
                0,
                1
            )

            # Calculate metrics
            malicious_precision = precision_score(
                y_validation,
                y_validation_pred,
                pos_label=1,
                zero_division=0
            )

            malicious_recall = recall_score(
                y_validation,
                y_validation_pred,
                pos_label=1,
                zero_division=0
            )

            malicious_f1 = f1_score(
                y_validation,
                y_validation_pred,
                pos_label=1,
                zero_division=0
            )

            # False Positive Rate
            cm = confusion_matrix(
                y_validation,
                y_validation_pred,
                labels=[0, 1]
            )

            tn, fp, fn, tp = cm.ravel()

            if (tn + fp) > 0:
                fpr = fp / (fp + tn)
            else:
                fpr = 0

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------
            #
            # We prioritize malicious detection while
            # penalizing false positives.
            #
            # 70% = malicious F1
            # 30% = low false-positive rate
            #
            score = (
                0.70 * malicious_f1
                + 0.30 * (1 - fpr)
            )

            search_results.append({
                "gamma": gamma,
                "nu": nu,
                "malicious_precision": malicious_precision,
                "malicious_recall": malicious_recall,
                "malicious_f1": malicious_f1,
                "fpr": fpr,
                "score": score
            })

            print(
                f"  Precision: {malicious_precision:.4f}"
            )

            print(
                f"  Recall   : {malicious_recall:.4f}"
            )

            print(
                f"  F1       : {malicious_f1:.4f}"
            )

            print(
                f"  FPR      : {fpr:.4f}"
            )

            print(
                f"  Score    : {score:.4f}"
            )

            print()

            # Select best model
            if score > best_score:

                best_score = score
                best_gamma = gamma
                best_nu = nu

        except Exception as e:

            print(
                f"  Error for gamma={gamma}, nu={nu}: {e}"
            )

            print()


# ============================================================
# 6. DISPLAY BEST PARAMETERS
# ============================================================

print("=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)

print("Best gamma:", best_gamma)
print("Best nu   :", best_nu)
print("Best score:", round(best_score, 4))


# ============================================================
# 7. TRAIN FINAL OSVM
# ============================================================

print()
print("=" * 60)
print("TRAINING FINAL ONE-CLASS SVM")
print("=" * 60)

# IMPORTANT:
# Final OSVM is trained ONLY using benign samples
# from the complete training dataset.

X_train_benign_final = X_train[
    y_train_binary == 0
]

print(
    "Final benign training samples:",
    len(X_train_benign_final)
)

print()
print("Training final One-Class SVM...")

final_model = OneClassSVM(
    kernel="rbf",
    gamma=best_gamma,
    nu=best_nu
)

final_model.fit(
    X_train_benign_final
)

print("Training completed.")


# ============================================================
# 8. PREDICT FINAL TEST SET
# ============================================================

print()
print("Predicting final test data...")

test_predictions = final_model.predict(
    X_test
)

# Convert OSVM output:
#
# +1 = Benign
# -1 = Malicious

y_pred = np.where(
    test_predictions == 1,
    0,
    1
)


# ============================================================
# 9. DISPLAY PREDICTION DISTRIBUTION
# ============================================================

predicted_labels = pd.Series(
    y_pred
).map({
    0: "Benign",
    1: "Malicious"
})

print()
print("Predicted labels:")
print(predicted_labels.value_counts())


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_binary,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print()
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_binary,
        y_pred,
        target_names=[
            "Benign",
            "Malicious"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. CALCULATE IMPORTANT METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_binary,
    y_pred
)

malicious_precision = precision_score(
    y_test_binary,
    y_pred,
    pos_label=1,
    zero_division=0
)

malicious_recall = recall_score(
    y_test_binary,
    y_pred,
    pos_label=1,
    zero_division=0
)

malicious_f1 = f1_score(
    y_test_binary,
    y_pred,
    pos_label=1,
    zero_division=0
)

if (tn + fp) > 0:
    fpr = fp / (fp + tn)
else:
    fpr = 0


print()
print("=" * 60)
print("FINAL OSVM METRICS")
print("=" * 60)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Malicious Precision: {malicious_precision * 100:.2f}%"
)

print(
    f"Malicious Recall   : {malicious_recall * 100:.2f}%"
)

print(
    f"Malicious F1 Score : {malicious_f1 * 100:.2f}%"
)

print(
    f"False Positive Rate: {fpr * 100:.2f}%"
)


# ============================================================
# 13. SAVE HYPERPARAMETER SEARCH RESULTS
# ============================================================

search_results_df = pd.DataFrame(
    search_results
)

search_results_df = search_results_df.sort_values(
    by="score",
    ascending=False
)

search_results_df.to_csv(
    "models/one_class_svm_search_results.csv",
    index=False
)

print()
print(
    "Hyperparameter search results saved to:"
)

print(
    "models/one_class_svm_search_results.csv"
)


# ============================================================
# 14. SAVE FINAL TEST PREDICTIONS
# ============================================================

results = X_test.copy()

results["actual_label"] = pd.Series(
    y_test_binary
).map({
    0: "Benign",
    1: "Malicious"
}).values

results["predicted_label"] = pd.Series(
    y_pred
).map({
    0: "Benign",
    1: "Malicious"
}).values

results.to_csv(
    "models/one_class_svm_results.csv",
    index=False
)

print()
print(
    "Final predictions saved to:"
)

print(
    "models/one_class_svm_results.csv"
)

print()
print("=" * 60)
print("ONE-CLASS SVM COMPLETED")
print("=" * 60)