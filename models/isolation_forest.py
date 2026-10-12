import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================

X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")

y_train = pd.read_csv("processed_data/y_train.csv")
y_test = pd.read_csv("processed_data/y_test.csv")

# Convert label columns to Series
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


# ==========================================
# 2. USE BENIGN TRAFFIC ONLY FOR TRAINING
# ==========================================

benign_mask = y_train.str.lower() == "benign"

X_train_benign = X_train[benign_mask]

print("\nBenign training samples:", len(X_train_benign))


# ==========================================
# 3. CREATE ISOLATION FOREST MODEL
# ==========================================

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining Isolation Forest...")

model.fit(X_train_benign)

print("Training completed.")


# ==========================================
# 5. PREDICT TEST DATA
# ==========================================

print("\nPredicting test data...")

predictions = model.predict(X_test)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

# Convert to project labels
y_pred = pd.Series(predictions).map({
    1: "Benign",
    -1: "Malicious"
})


# ==========================================
# 6. DISPLAY PREDICTION COUNTS
# ==========================================

print("\nPredicted labels:")
print(y_pred.value_counts())


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Benign", "Malicious"]
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Benign", "Malicious"],
        target_names=["Benign", "Malicious"],
        zero_division=0
    )
)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv(
    "isolation_forest_results.csv",
    index=False
)

print("\nResults saved to:")
print("isolation_forest_results.csv")