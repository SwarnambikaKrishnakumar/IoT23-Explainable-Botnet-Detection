import zipfile
import pandas as pd

# Location of your IoT-23 ZIP file
ZIP_PATH = r"C:\Users\shakt\Downloads\archive (5).zip"

# Files we selected for the project
SELECTED_FILES = [
    "dataset1.csv",
    "dataset5.csv",
    "dataset11.csv",
    "dataset13.csv"
]

dataframes = []

# Open the ZIP file without extracting it
with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for filename in SELECTED_FILES:

        print("Reading:", filename)

        # Read CSV directly from the ZIP
        df = pd.read_csv(
            z.open(filename),
            low_memory=False
        )

        print("Rows:", len(df))
        print("Columns before correction:", len(df.columns))

        # The last physical column contains:
        # tunnel_parents, label, detailed-label
        combined = df.iloc[:, -1].astype(str).str.split()

        df["tunnel_parents"] = combined.str[0]
        df["label"] = combined.str[1]
        df["detailed-label"] = combined.str[2]

        # Remove the original combined column
        df = df.drop(columns=[df.columns[-4]])

        print("Columns after correction:", len(df.columns))
        print()

        dataframes.append(df)


# Combine all selected datasets
combined_df = pd.concat(
    dataframes,
    ignore_index=True
)

print("===================================")
print("COMBINED DATASET")
print("===================================")

print("Total rows:", len(combined_df))
print("Total columns:", len(combined_df.columns))

# Normalize labels
combined_df["label"] = (
    combined_df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

combined_df["label"] = combined_df["label"].replace({
    "benign": "Benign",
    "malicious": "Malicious"
})

print("\nLABEL DISTRIBUTION")
print("------------------")
print(combined_df["label"].value_counts())

print("\nDETAILED ATTACK LABELS")
print("----------------------")
print(combined_df["detailed-label"].value_counts())

print("\nMISSING VALUES")
print("--------------")
print(combined_df.isnull().sum())

print("\nDUPLICATES")
print("----------")
before = len(combined_df)

combined_df = combined_df.drop_duplicates()

after = len(combined_df)

print("Duplicates removed:", before - after)
print("Remaining rows:", after)

print("\nPREPROCESSING CHECK COMPLETED")
# ==========================================
# DATA CLEANING
# ==========================================

print("\n===================================")
print("DATA CLEANING")
print("===================================")

# Columns that should contain numerical values
numeric_columns = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes"
]

# Convert '-' to missing value ONLY in numerical columns
for column in numeric_columns:
    combined_df[column] = pd.to_numeric(
        combined_df[column].replace("-", pd.NA),
        errors="coerce"
    )

print("\nNumeric columns converted.")

# Show missing values created during conversion
print("\nMissing values after numeric conversion:")
print(combined_df[numeric_columns].isnull().sum())
# Handle missing values in network-flow measurements
flow_columns = [
    "duration",
    "orig_bytes",
    "resp_bytes"
]

combined_df[flow_columns] = combined_df[flow_columns].fillna(0)

print("\nMissing flow values replaced with 0.")

print("\nMissing values after imputation:")
print(combined_df[flow_columns].isnull().sum())
# ==========================================
# CATEGORICAL DATA CLEANING
# ==========================================

categorical_columns = [
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
    "tunnel_parents"
]

# Replace '-' with "unknown"
for column in categorical_columns:
    combined_df[column] = combined_df[column].replace("-", "unknown")

print("\nCategorical '-' values replaced with 'unknown'.")

print("\nUnique values in categorical columns:")
for column in categorical_columns:
    print(f"\n{column}:")
    print(combined_df[column].value_counts().head(10))
    # ==========================================
# FEATURE SELECTION
# ==========================================

print("\n===================================")
print("FEATURE SELECTION")
print("===================================")

# Remove columns that are identifiers,
# unavailable in our data, or contain target information
columns_to_remove = [
    "uid",
    "id.orig_h",
    "id.resp_h",
    "local_orig",
    "local_resp",
    "tunnel_parents",
    "label",
    "detailed-label"
]

model_df = combined_df.drop(
    columns=columns_to_remove
)

print("\nColumns retained for model preparation:")
print(model_df.columns.tolist())

print("\nNumber of model features before encoding:",
      len(model_df.columns))
# ==========================================
# PREPARE FEATURES AND TARGET
# ==========================================

print("\n===================================")
print("PREPARING FEATURES AND TARGET")
print("===================================")

# Remove timestamp from model features
model_df = model_df.drop(columns=["ts"])

# Target variable
y = combined_df["label"]

# Feature matrix
X = model_df.copy()

print("\nFeatures used for the model:")
print(X.columns.tolist())

print("\nNumber of features before encoding:", X.shape[1])

print("\nTarget distribution:")
print(y.value_counts())
# Identify categorical features
categorical_features = [
    "proto",
    "service",
    "conn_state",
    "history"
]

print("\nCategorical features:")
print(categorical_features)
# ==========================================
# CATEGORICAL ENCODING
# ==========================================

from sklearn.preprocessing import OneHotEncoder

print("\n===================================")
print("CATEGORICAL ENCODING")
print("===================================")

categorical_features = [
    "proto",
    "service",
    "conn_state",
    "history"
]

# Create One-Hot Encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Encode categorical features
encoded_array = encoder.fit_transform(
    X[categorical_features]
)

# Get generated feature names
encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
)

# Create DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoded_feature_names,
    index=X.index
)

# Remove original categorical columns
X_numeric = X.drop(
    columns=categorical_features
)

# Combine numerical and encoded categorical features
X_encoded = pd.concat(
    [X_numeric, encoded_df],
    axis=1
)

print("\nOriginal feature count:", X.shape[1])
print("Encoded feature count:", X_encoded.shape[1])

print("\nEncoded feature names:")
print(X_encoded.columns.tolist())
# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

print("\n===================================")
print("TRAIN-TEST SPLIT")
print("===================================")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
# ==========================================
# FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

print("\n===================================")
print("FEATURE SCALING")
print("===================================")

# Numerical features
numerical_features = [
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes"
]

# Create scaler
scaler = StandardScaler()

# Fit ONLY on training numerical features
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

# Use the already-fitted scaler on test data
X_test_scaled[numerical_features] = scaler.transform(
    X_test[numerical_features]
)

print("\nScaling completed successfully.")

print("\nTraining data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\nTraining numerical feature means after scaling:")
print(X_train_scaled[numerical_features].mean().round(3))

print("\nTraining numerical feature standard deviations:")
print(X_train_scaled[numerical_features].std().round(3))
# ==========================================
# SAVE PREPROCESSED DATA
# ==========================================

import os

OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train_scaled.to_csv(
    os.path.join(OUTPUT_DIR, "X_train.csv"),
    index=False
)

X_test_scaled.to_csv(
    os.path.join(OUTPUT_DIR, "X_test.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(OUTPUT_DIR, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(OUTPUT_DIR, "y_test.csv"),
    index=False
)

print("\n===================================")
print("PREPROCESSED DATA SAVED")
print("===================================")

print("Saved to:", os.path.abspath(OUTPUT_DIR))
print("\n===== SAMPLE CORRECTED DATASET =====")
print(df.head(10).to_string(index=False))