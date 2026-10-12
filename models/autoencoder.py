import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")

y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

# Convert labels
# Benign = 0
# Malicious = 1
y_train = np.where(y_train == "Malicious", 1, 0)
y_test = np.where(y_test == "Malicious", 1, 0)

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

print("\nTraining labels:")
print(pd.Series(y_train).value_counts())

print("\nTesting labels:")
print(pd.Series(y_test).value_counts())


# ============================================================
# 2. SELECT ONLY BENIGN TRAFFIC FOR TRAINING
# ============================================================

X_train_benign = X_train[y_train == 0]

print("\nBenign training samples:", len(X_train_benign))


# ============================================================
# 3. BUILD AUTOENCODER
# ============================================================

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)

# Decoder
decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("\nAutoencoder architecture:")
autoencoder.summary()


# ============================================================
# 4. TRAIN AUTOENCODER USING ONLY BENIGN TRAFFIC
# ============================================================

print("\nTraining Autoencoder...")

history = autoencoder.fit(
    X_train_benign,
    X_train_benign,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

print("\nTraining completed.")


# ============================================================
# 5. CALCULATE RECONSTRUCTION ERROR
# ============================================================

print("\nCalculating reconstruction errors...")

X_test_pred = autoencoder.predict(X_test, verbose=0)

reconstruction_error = np.mean(
    np.square(X_test - X_test_pred),
    axis=1
)


# ============================================================
# 6. DETERMINE ANOMALY THRESHOLD
# ============================================================

# Calculate reconstruction error for benign training traffic

X_benign_pred = autoencoder.predict(
    X_train_benign,
    verbose=0
)

benign_error = np.mean(
    np.square(X_train_benign - X_benign_pred),
    axis=1
)

# Use the 95th percentile as threshold
threshold = np.percentile(benign_error, 95)

print("\nAnomaly threshold:", threshold)


# ============================================================
# 7. PREDICT ANOMALIES
# ============================================================

# Error greater than threshold = malicious/anomaly
y_pred = (reconstruction_error > threshold).astype(int)


print("\nPredicted labels:")
print(
    pd.Series(
        np.where(y_pred == 1, "Malicious", "Benign")
    ).value_counts()
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malicious"],
        digits=2
    )
)


# ============================================================
# 10. CALCULATE FINAL METRICS
# ============================================================

accuracy = (tp + tn) / (tp + tn + fp + fn)

malicious_precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

malicious_recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

malicious_f1 = (
    2 * malicious_precision * malicious_recall /
    (malicious_precision + malicious_recall)
    if (malicious_precision + malicious_recall) > 0
    else 0
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)


print("\n========================================")
print("FINAL AUTOENCODER METRICS")
print("========================================")

print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Malicious Precision: {malicious_precision * 100:.2f}%")
print(f"Malicious Recall   : {malicious_recall * 100:.2f}%")
print(f"Malicious F1 Score : {malicious_f1 * 100:.2f}%")
print(f"False Positive Rate: {false_positive_rate * 100:.2f}%")


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Actual": np.where(y_test == 1, "Malicious", "Benign"),
    "Predicted": np.where(y_pred == 1, "Malicious", "Benign"),
    "Reconstruction_Error": reconstruction_error
})

results.to_csv(
    "models/autoencoder_results.csv",
    index=False
)

print("\nResults saved to:")
print("models/autoencoder_results.csv")