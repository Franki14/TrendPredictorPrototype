"""
Train and Save Final Model

Trains the established Balanced Random Forest using the frozen
training dataset and saves the fitted estimator for use by the
Trend Predictor prototype.

No validation or test-set information is used in this script.
"""

import os
import json
import hashlib

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/processed/cascade_train_network.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_random_forest.joblib",
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json",
)

TARGET = "Target"

FEATURES = [
    "TimeTo5",
    "MinInterarrival",
    "EarlyAcceleration",
    "NeighbourhoodReach",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]

RANDOM_STATE = 42
N_ESTIMATORS = 500


# ============================================================
# Dataset Hash
# ============================================================

def calculate_sha256(filepath):
    """
    Calculate SHA-256 hash for reproducibility.
    """

    sha256 = hashlib.sha256()

    with open(filepath, "rb") as file:
        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Load Training Data
# ============================================================

def load_training_data():

    print("Loading frozen training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    print(
        f"Training shape: {train_df.shape}"
    )

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{missing_columns}"
        )

    if train_df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values detected in model features or target."
        )

    print("Dataset validation: PASSED")

    return train_df


# ============================================================
# Build Frozen Model
# ============================================================

def build_final_model():
    """
    Return the established final Balanced Random Forest.

    These parameters were frozen before final test evaluation.
    """

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Save Model
# ============================================================

def save_model(model):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Saved model: {MODEL_PATH}"
    )


# ============================================================
# Save Metadata
# ============================================================

def save_metadata(train_df):

    metadata = {
        "model_name": "Balanced Random Forest",
        "model_file": "final_random_forest.joblib",

        "model_parameters": {
            "n_estimators": N_ESTIMATORS,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },

        "classification_threshold": 0.5,

        "target": TARGET,

        "features": FEATURES,

        "number_of_features": len(FEATURES),

        "training_rows": len(train_df),

        "training_class_distribution": {
            str(key): int(value)
            for key, value
            in train_df[TARGET].value_counts().to_dict().items()
        },

        "training_dataset": TRAIN_PATH,

        "training_dataset_sha256": calculate_sha256(
            TRAIN_PATH
        ),

        "score_definition": (
            "Viral Potential Score (VPS) is derived from the "
            "Random Forest predicted probability for the positive "
            "trend-emergence class, multiplied by 100."
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"Saved metadata: {METADATA_PATH}"
    )


# ============================================================
# Verify Saved Model
# ============================================================

def verify_saved_model(
    X_train,
):

    print("\nVerifying saved model...")

    loaded_model = joblib.load(
        MODEL_PATH
    )

    sample = X_train.iloc[[0]]

    probability = loaded_model.predict_proba(
        sample
    )[0, 1]

    prediction = loaded_model.predict(
        sample
    )[0]

    print(
        "Model reload: PASSED"
    )

    print(
        f"Example prediction: {prediction}"
    )

    print(
        f"Example probability: {probability:.4f}"
    )

    print(
        f"Example VPS: {probability * 100:.2f}/100"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "TRAIN AND SAVE FINAL MODEL"
    )

    print(
        "#" * 80
    )

    train_df = load_training_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    print(
        "\nTraining class distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nFinal model configuration:"
    )

    print(
        "Model:        Balanced Random Forest"
    )

    print(
        f"Features:     {len(FEATURES)}"
    )

    print(
        f"n_estimators: {N_ESTIMATORS}"
    )

    print(
        "class_weight: balanced"
    )

    print(
        f"random_state: {RANDOM_STATE}"
    )

    # ========================================================
    # Train
    # ========================================================

    model = build_final_model()

    print(
        "\nTraining final model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # Save
    # ========================================================

    save_model(
        model
    )

    save_metadata(
        train_df
    )

    # ========================================================
    # Verify
    # ========================================================

    verify_saved_model(
        X_train
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL MODEL ARTIFACT CREATED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()