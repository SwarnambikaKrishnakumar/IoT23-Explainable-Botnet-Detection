import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, Sequential
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
y_train = np.where(y_train == "Malicious", 1, 0)
y_test = np.where(y_test == "Malicious", 1, 0)

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

print("\nTraining labels:")
print(pd.Series(y_train).value_counts())

print("\nTesting labels:")
print(pd.Series(y_test).value_counts())


# ============================================================
# 2. USE ONLY BENIGN TRAFFIC
# ============================================================

X_train_benign = X_train[y_train == 0].values.astype("float32")

X_test_np = X_test.values.astype("float32")

print("\nBenign training samples:", len(X_train_benign))


# ============================================================
# 3. GENERATOR
# ============================================================

input_dim = X_train.shape[1]
latent_dim = 16

generator = Sequential([
    Dense(32, activation="relu", input_dim=latent_dim),
    Dense(56, activation="linear")
])

print("\nGenerator created.")


# ============================================================
# 4. DISCRIMINATOR
# ============================================================

discriminator = Sequential([
    Dense(32, activation="relu", input_dim=input_dim),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

print("Discriminator created.")


# ============================================================
# 5. COMBINED GAN
# ============================================================

discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)

gan = Model(gan_input, gan_output)

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

print("GAN created.")


# ============================================================
# 6. TRAIN GAN
# ============================================================

epochs = 30
batch_size = 128

print("\nTraining GAN...")

for epoch in range(epochs):

    # --------------------------------------------------------
    # Select real benign samples
    # --------------------------------------------------------

    idx = np.random.randint(
        0,
        len(X_train_benign),
        batch_size
    )

    real_samples = X_train_benign[idx]

    # --------------------------------------------------------
    # Generate fake samples
    # --------------------------------------------------------

    noise = np.random.normal(
        0,
        1,
        (batch_size, latent_dim)
    ).astype("float32")

    fake_samples = generator.predict(
        noise,
        verbose=0
    )

    # --------------------------------------------------------
    # Train discriminator
    # --------------------------------------------------------

    discriminator.trainable = True

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(
        real_samples,
        real_labels
    )

    d_loss_fake = discriminator.train_on_batch(
        fake_samples,
        fake_labels
    )

    d_loss = (d_loss_real + d_loss_fake) / 2

    # --------------------------------------------------------
    # Train generator
    # --------------------------------------------------------

    discriminator.trainable = False

    noise = np.random.normal(
        0,
        1,
        (batch_size, latent_dim)
    ).astype("float32")

    valid_labels = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(
        noise,
        valid_labels
    )

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- D Loss: {d_loss:.4f} "
            f"- G Loss: {float(g_loss):.4f}"
        )


print("\nGAN training completed.")


# ============================================================
# 7. GENERATE NORMAL REFERENCE SAMPLES
# ============================================================

print("\nGenerating normal reference samples...")

noise = np.random.normal(
    0,
    1,
    (len(X_test_np), latent_dim)
).astype("float32")

generated_samples = generator.predict(
    noise,
    verbose=0
)


# ============================================================
# 8. CALCULATE ANOMALY SCORE
# ============================================================

# Distance between test traffic and generated normal pattern

anomaly_score = np.mean(
    np.square(
        X_test_np - generated_samples
    ),
    axis=1
)


# ============================================================
# 9. DETERMINE THRESHOLD USING BENIGN TRAFFIC
# ============================================================

normal_noise = np.random.normal(
    0,
    1,
    (len(X_train_benign), latent_dim)
).astype("float32")

normal_generated = generator.predict(
    normal_noise,
    verbose=0
)

normal_error = np.mean(
    np.square(
        X_train_benign - normal_generated
    ),
    axis=1
)

threshold = np.percentile(
    normal_error,
    95
)

print("\nAnomaly threshold:", threshold)


# ============================================================
# 10. PREDICT
# ============================================================

y_pred = (
    anomaly_score > threshold
).astype(int)

print("\nPredicted labels:")

print(
    pd.Series(
        np.where(
            y_pred == 1,
            "Malicious",
            "Benign"
        )
    ).value_counts()
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Benign",
            "Malicious"
        ],
        digits=2
    )
)


# ============================================================
# 13. FINAL METRICS
# ============================================================

accuracy = (
    (tp + tn) /
    (tp + tn + fp + fn)
)

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
    2 *
    malicious_precision *
    malicious_recall /
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
print("FINAL GAN METRICS")
print("========================================")

print(
    f"Accuracy           : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Malicious Precision: "
    f"{malicious_precision * 100:.2f}%"
)

print(
    f"Malicious Recall   : "
    f"{malicious_recall * 100:.2f}%"
)

print(
    f"Malicious F1 Score : "
    f"{malicious_f1 * 100:.2f}%"
)

print(
    f"False Positive Rate: "
    f"{false_positive_rate * 100:.2f}%"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Actual": np.where(
        y_test == 1,
        "Malicious",
        "Benign"
    ),

    "Predicted": np.where(
        y_pred == 1,
        "Malicious",
        "Benign"
    ),

    "Anomaly_Score": anomaly_score
})

results.to_csv(
    "models/gan_results.csv",
    index=False
)

print("\nResults saved to:")
print("models/gan_results.csv")