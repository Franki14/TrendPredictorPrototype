from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OBSERVATION_SIZE = 5
TARGET_THRESHOLD = 34

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "twitter"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------
# Cascade parsing
# ---------------------------------------------------------

def parse_cascade_line(line):
    """
    Parse one processed DisenIDP cascade.

    Expected format:
        user_id,timestamp user_id,timestamp ...

    Returns
    -------
    list of tuples
        [(user_id, timestamp), ...]
    """

    events = []

    for item in line.strip().split():
        try:
            user_id, timestamp = item.split(",")
            events.append((int(user_id), float(timestamp)))
        except (ValueError, IndexError):
            continue

    # Ensure events are chronologically ordered.
    events.sort(key=lambda x: x[1])

    return events


# ---------------------------------------------------------
# Temporal feature extraction
# ---------------------------------------------------------

def extract_temporal_features(events):
    """
    Extract features using ONLY the first five cascade events.
    """

    early_events = events[:OBSERVATION_SIZE]

    timestamps = np.array(
        [timestamp for _, timestamp in early_events],
        dtype=float
    )

    # Time differences between consecutive adopters.
    interarrival = np.diff(timestamps)

    time_to_5 = timestamps[-1] - timestamps[0]

    mean_interarrival = np.mean(interarrival)
    std_interarrival = np.std(interarrival)
    min_interarrival = np.min(interarrival)
    max_interarrival = np.max(interarrival)

    # Number of new adopters per second during the early window.
    if time_to_5 > 0:
        early_velocity = (OBSERVATION_SIZE - 1) / time_to_5
    else:
        early_velocity = 0.0

    # Compare the first half and second half of early inter-arrival times.
    #
    # Smaller later gaps indicate acceleration.
    first_half = np.mean(interarrival[:2])
    second_half = np.mean(interarrival[2:])

    early_acceleration = first_half - second_half

    return {
        "TimeTo5": time_to_5,
        "MeanInterarrival": mean_interarrival,
        "StdInterarrival": std_interarrival,
        "MinInterarrival": min_interarrival,
        "MaxInterarrival": max_interarrival,
        "EarlyVelocity": early_velocity,
        "EarlyAcceleration": early_acceleration,
    }


# ---------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------

def process_split(filename, split_name):
    """
    Convert a cascade split into one row per eligible cascade.
    """

    filepath = RAW_DATA_DIR / filename

    rows = []

    with open(filepath, "r") as file:

        for cascade_id, line in enumerate(file):

            events = parse_cascade_line(line)

            final_size = len(events)

            # We need at least five events to form the observation window.
            if final_size < OBSERVATION_SIZE:
                continue

            features = extract_temporal_features(events)

            target = int(final_size >= TARGET_THRESHOLD)

            row = {
                "CascadeID": cascade_id,
                "Split": split_name,
                "FinalSize": final_size,
                "Target": target,
                **features,
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_dataset(df):
    """
    Run basic checks against leakage and preprocessing errors.
    """

    assert df["Target"].isin([0, 1]).all()

    assert (df["FinalSize"] >= OBSERVATION_SIZE).all()

    temporal_columns = [
        "TimeTo5",
        "MeanInterarrival",
        "StdInterarrival",
        "MinInterarrival",
        "MaxInterarrival",
        "EarlyVelocity",
        "EarlyAcceleration",
    ]

    assert df[temporal_columns].notna().all().all()

    print("\nClass distribution:")

    print(
        df["Target"]
        .value_counts()
        .sort_index()
    )

    print("\nClass proportions:")

    print(
        df["Target"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = process_split(
        "cascadetrain.txt",
        "train"
    )

    valid_df = process_split(
        "cascadevalid.txt",
        "validation"
    )

    test_df = process_split(
        "cascadetest.txt",
        "test"
    )

    datasets = {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }

    for name, df in datasets.items():

        print("\n" + "=" * 60)
        print(name.upper())
        print("=" * 60)

        print("Shape:", df.shape)

        validate_dataset(df)

        output_path = OUTPUT_DIR / f"cascade_{name}.csv"

        df.to_csv(
            output_path,
            index=False
        )

        print("\nSaved:", output_path)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()