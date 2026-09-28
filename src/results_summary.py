"""
Consolidated Experimental Results

This script consolidates the main experimental results obtained during
model development and final evaluation.

IMPORTANT:
- No models are trained in this script.
- No hyperparameters are selected in this script.
- The held-out test set is not used for model selection.
- Validation and test results are reported separately.
"""

import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


# ============================================================
# Main Validation Model Results
# ============================================================

def create_validation_model_summary():
    """
    Consolidate the principal validation-set model results.

    These results were obtained during model development.
    """

    results = [
        {
            "Model": "Most Frequent Baseline",
            "Accuracy": 0.7067,
            "Precision": 0.0000,
            "Recall": 0.0000,
            "F1": 0.0000,
            "ROC_AUC": 0.5000,
            "PR_AUC": 0.2933,
        },
        {
            "Model": "Stratified Random Baseline",
            "Accuracy": 0.5618,
            "Precision": 0.2588,
            "Recall": 0.2651,
            "F1": 0.2619,
            "ROC_AUC": 0.4750,
            "PR_AUC": 0.2842,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Accuracy": 0.5477,
            "Precision": 0.3869,
            "Recall": 0.9277,
            "F1": 0.5461,
            "ROC_AUC": 0.6513,
            "PR_AUC": 0.3789,
        },
        {
            "Model": "Balanced Random Forest",
            "Accuracy": 0.7809,
            "Precision": 0.5913,
            "Recall": 0.8193,
            "F1": 0.6869,
            "ROC_AUC": 0.8584,
            "PR_AUC": 0.7287,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Random Forest Class-Weight Experiment
# ============================================================

def create_class_weight_summary():
    """
    Results from the Random Forest class-weight experiment.

    This experiment compared the unweighted and balanced
    Random Forest configurations on the validation set.
    """

    results = [
        {
            "Model": "Unweighted Random Forest",
            "Accuracy": 0.7880,
            "Precision": 0.6353,
            "Recall": 0.6506,
            "F1": 0.6429,
            "ROC_AUC": 0.8580,
            "PR_AUC": 0.7192,
            "TN": 169,
            "FP": 31,
            "FN": 29,
            "TP": 54,
        },
        {
            "Model": "Balanced Random Forest",
            "Accuracy": 0.7809,
            "Precision": 0.5913,
            "Recall": 0.8193,
            "F1": 0.6869,
            "ROC_AUC": 0.8584,
            "PR_AUC": 0.7287,
            "TN": 153,
            "FP": 47,
            "FN": 15,
            "TP": 68,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Feature-Group Ablation Results
# ============================================================

def create_ablation_summary():
    """
    Results from the temporal/network feature-group ablation study.
    """

    results = [
        {
            "FeatureGroup": "Temporal Only",
            "NumFeatures": 3,
            "Accuracy": 0.5265,
            "Precision": 0.3023,
            "Recall": 0.4699,
            "F1": 0.3679,
            "ROC_AUC": 0.5402,
            "PR_AUC": 0.3377,
            "TN": 110,
            "FP": 90,
            "FN": 44,
            "TP": 39,
        },
        {
            "FeatureGroup": "Network Only",
            "NumFeatures": 5,
            "Accuracy": 0.7456,
            "Precision": 0.5647,
            "Recall": 0.5783,
            "F1": 0.5714,
            "ROC_AUC": 0.7758,
            "PR_AUC": 0.5909,
            "TN": 163,
            "FP": 37,
            "FN": 35,
            "TP": 48,
        },
        {
            "FeatureGroup": "Combined",
            "NumFeatures": 8,
            "Accuracy": 0.7809,
            "Precision": 0.5913,
            "Recall": 0.8193,
            "F1": 0.6869,
            "ROC_AUC": 0.8584,
            "PR_AUC": 0.7287,
            "TN": 153,
            "FP": 47,
            "FN": 15,
            "TP": 68,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Hyperparameter Experiment Summary
# ============================================================

def create_tuning_summary():
    """
    Summarise the main Random Forest configuration and the
    validation configuration selected by the tuning experiment.

    The tuned configuration achieved a slightly higher ROC-AUC
    but did not replace the frozen main Random Forest model.
    """

    results = [
        {
            "Configuration": "Main Balanced RF",
            "MaxDepth": "None",
            "MinSamplesLeaf": 1,
            "MaxFeatures": "sqrt",
            "Accuracy": 0.7809,
            "Precision": 0.5913,
            "Recall": 0.8193,
            "F1": 0.6869,
            "ROC_AUC": 0.8584,
            "PR_AUC": 0.7287,
        },
        {
            "Configuration": "Tuning Experiment Selection",
            "MaxDepth": "20",
            "MinSamplesLeaf": 1,
            "MaxFeatures": "sqrt",
            "Accuracy": 0.7703,
            "Precision": 0.5776,
            "Recall": 0.8072,
            "F1": 0.6734,
            "ROC_AUC": 0.8620,
            "PR_AUC": 0.7337,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Bootstrap Confidence Intervals — Validation
# ============================================================

def create_bootstrap_summary():
    """
    Validation performance with 95% bootstrap confidence intervals.
    """

    results = [
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "Accuracy",
            "Estimate": 0.5477,
            "CI_Lower": 0.4876,
            "CI_Upper": 0.6042,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "Precision",
            "Estimate": 0.3869,
            "CI_Lower": 0.3200,
            "CI_Upper": 0.4545,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "Recall",
            "Estimate": 0.9277,
            "CI_Lower": 0.8684,
            "CI_Upper": 0.9756,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "F1",
            "Estimate": 0.5461,
            "CI_Lower": 0.4737,
            "CI_Upper": 0.6118,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "ROC_AUC",
            "Estimate": 0.6513,
            "CI_Lower": 0.5844,
            "CI_Upper": 0.7174,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Metric": "PR_AUC",
            "Estimate": 0.3789,
            "CI_Lower": 0.3025,
            "CI_Upper": 0.4819,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "Accuracy",
            "Estimate": 0.7809,
            "CI_Lower": 0.7314,
            "CI_Upper": 0.8269,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "Precision",
            "Estimate": 0.5913,
            "CI_Lower": 0.5000,
            "CI_Upper": 0.6783,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "Recall",
            "Estimate": 0.8193,
            "CI_Lower": 0.7333,
            "CI_Upper": 0.8989,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "F1",
            "Estimate": 0.6869,
            "CI_Lower": 0.6070,
            "CI_Upper": 0.7562,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "ROC_AUC",
            "Estimate": 0.8584,
            "CI_Lower": 0.8122,
            "CI_Upper": 0.9019,
        },
        {
            "Model": "Balanced Random Forest",
            "Metric": "PR_AUC",
            "Estimate": 0.7287,
            "CI_Lower": 0.6401,
            "CI_Upper": 0.8107,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Paired Bootstrap Comparison
# ============================================================

def create_paired_bootstrap_summary():
    """
    Paired bootstrap comparison:

    Balanced Random Forest minus Balanced Logistic Regression.
    """

    results = [
        {
            "Metric": "Accuracy",
            "RF_minus_LR": 0.2332,
            "CI_Lower": 0.1696,
            "CI_Upper": 0.2968,
            "P_Value": 0.0000,
            "CI_Excludes_Zero": True,
        },
        {
            "Metric": "Precision",
            "RF_minus_LR": 0.2044,
            "CI_Lower": 0.1465,
            "CI_Upper": 0.2684,
            "P_Value": 0.0000,
            "CI_Excludes_Zero": True,
        },
        {
            "Metric": "Recall",
            "RF_minus_LR": -0.1084,
            "CI_Lower": -0.1972,
            "CI_Upper": -0.0222,
            "P_Value": 0.0200,
            "CI_Excludes_Zero": True,
        },
        {
            "Metric": "F1",
            "RF_minus_LR": 0.1408,
            "CI_Lower": 0.0773,
            "CI_Upper": 0.2048,
            "P_Value": 0.0000,
            "CI_Excludes_Zero": True,
        },
        {
            "Metric": "ROC_AUC",
            "RF_minus_LR": 0.2071,
            "CI_Lower": 0.1353,
            "CI_Upper": 0.2813,
            "P_Value": 0.0000,
            "CI_Excludes_Zero": True,
        },
        {
            "Metric": "PR_AUC",
            "RF_minus_LR": 0.3498,
            "CI_Lower": 0.2398,
            "CI_Upper": 0.4350,
            "P_Value": 0.0000,
            "CI_Excludes_Zero": True,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# McNemar Test Summary
# ============================================================

def create_mcnemar_summary():
    """
    Exact McNemar test comparing paired classification errors
    between Logistic Regression and Random Forest.
    """

    results = [
        {
            "LR_Correct_RF_Wrong": 16,
            "LR_Wrong_RF_Correct": 82,
            "Statistic": 16.0000,
            "P_Value": 0.000000,
            "Significant_At_0.05": True,
        }
    ]

    return pd.DataFrame(results)


# ============================================================
# Final Held-Out Test Results
# ============================================================

def create_test_summary():
    """
    Final reproducible evaluation of the frozen Balanced Random Forest
    on the held-out test set.

    The test set was not used for model selection.
    """

    results = [
        {
            "Model": "Final Balanced Random Forest",
            "Dataset": "Held-Out Test",
            "Accuracy": 0.7912,
            "Precision": 0.4643,
            "Recall": 0.6964,
            "F1": 0.5571,
            "ROC_AUC": 0.8257,
            "PR_AUC": 0.6084,
            "TN": 196,
            "FP": 45,
            "FN": 17,
            "TP": 39,
        }
    ]

    return pd.DataFrame(results)


# ============================================================
# Final Test Confidence Intervals
# ============================================================

def create_test_ci_summary():
    """
    95% bootstrap confidence intervals for the final reproducible
    held-out test evaluation.
    """

    results = [
        {
            "Metric": "Accuracy",
            "Estimate": 0.7912,
            "CI_Lower": 0.7441,
            "CI_Upper": 0.8384,
        },
        {
            "Metric": "Precision",
            "Estimate": 0.4643,
            "CI_Lower": 0.3611,
            "CI_Upper": 0.5699,
        },
        {
            "Metric": "Recall",
            "Estimate": 0.6964,
            "CI_Lower": 0.5692,
            "CI_Upper": 0.8182,
        },
        {
            "Metric": "F1",
            "Estimate": 0.5571,
            "CI_Lower": 0.4545,
            "CI_Upper": 0.6490,
        },
        {
            "Metric": "ROC_AUC",
            "Estimate": 0.8257,
            "CI_Lower": 0.7598,
            "CI_Upper": 0.8865,
        },
        {
            "Metric": "PR_AUC",
            "Estimate": 0.6084,
            "CI_Lower": 0.4784,
            "CI_Upper": 0.7261,
        },
    ]

    return pd.DataFrame(results)


# ============================================================
# Save Table
# ============================================================

def save_table(dataframe, filename):

    os.makedirs(
        TABLES_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        TABLES_DIR,
        filename,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Print Table
# ============================================================

def print_table(title, dataframe):

    print(
        "\n" + "=" * 100
    )

    print(title)

    print(
        "=" * 100
    )

    print(
        dataframe.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 100
    )

    print(
        "CONSOLIDATED EXPERIMENTAL RESULTS"
    )

    print(
        "#" * 100
    )

    # ========================================================
    # 11.1 Main Validation Results
    # ========================================================

    validation_df = (
        create_validation_model_summary()
    )

    print_table(
        "MAIN MODEL COMPARISON — VALIDATION SET",
        validation_df,
    )

    save_table(
        validation_df,
        "validation_model_comparison.csv",
    )

    # ========================================================
    # Random Forest Class Weight Experiment
    # ========================================================

    class_weight_df = (
        create_class_weight_summary()
    )

    print_table(
        "RANDOM FOREST CLASS-WEIGHT EXPERIMENT",
        class_weight_df,
    )

    save_table(
        class_weight_df,
        "random_forest_class_weight_comparison.csv",
    )

    # ========================================================
    # Feature Ablation
    # ========================================================

    ablation_df = (
        create_ablation_summary()
    )

    print_table(
        "FEATURE-GROUP ABLATION STUDY",
        ablation_df,
    )

    save_table(
        ablation_df,
        "feature_group_ablation.csv",
    )

    # ========================================================
    # Hyperparameter Experiment
    # ========================================================

    tuning_df = (
        create_tuning_summary()
    )

    print_table(
        "RANDOM FOREST TUNING SUMMARY",
        tuning_df,
    )

    save_table(
        tuning_df,
        "random_forest_tuning_summary.csv",
    )

    # ========================================================
    # Bootstrap Confidence Intervals
    # ========================================================

    bootstrap_df = (
        create_bootstrap_summary()
    )

    print_table(
        "VALIDATION BOOTSTRAP CONFIDENCE INTERVALS",
        bootstrap_df,
    )

    save_table(
        bootstrap_df,
        "validation_bootstrap_confidence_intervals.csv",
    )

    # ========================================================
    # Paired Bootstrap
    # ========================================================

    paired_df = (
        create_paired_bootstrap_summary()
    )

    print_table(
        "PAIRED BOOTSTRAP — RF MINUS LR",
        paired_df,
    )

    save_table(
        paired_df,
        "paired_bootstrap_comparison.csv",
    )

    # ========================================================
    # McNemar
    # ========================================================

    mcnemar_df = (
        create_mcnemar_summary()
    )

    print_table(
        "MCNEMAR TEST — LR VS RF",
        mcnemar_df,
    )

    save_table(
        mcnemar_df,
        "mcnemar_test.csv",
    )

    # ========================================================
    # Final Held-Out Test
    # ========================================================

    test_df = (
        create_test_summary()
    )

    print_table(
        "FINAL FROZEN MODEL — HELD-OUT TEST SET",
        test_df,
    )

    save_table(
        test_df,
        "final_test_results.csv",
    )

    # ========================================================
    # Final Test Confidence Intervals
    # ========================================================

    test_ci_df = (
        create_test_ci_summary()
    )

    print_table(
        "FINAL TEST — 95% BOOTSTRAP CONFIDENCE INTERVALS",
        test_ci_df,
    )

    save_table(
        test_ci_df,
        "final_test_confidence_intervals.csv",
    )

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "RESULT CONSOLIDATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nTables saved to: {TABLES_DIR}/"
    )


if __name__ == "__main__":
    main()