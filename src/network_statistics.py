from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cascade_train_network.csv"
)


NETWORK_FEATURES = [
    "MeanDegree",
    "MaxDegree",
    "DegreeStd",
    "EarlyInternalEdges",
    "EarlyDensity",
    "NeighbourhoodReach",
    "MeanClustering",
    "EarlyCommunityCount",
    "CommunityDiversity",
    "MeanPageRank",
    "MaxPageRank",
]


def rank_biserial_from_u(u, n0, n1):
    """
    Rank-biserial correlation derived from the
    Mann-Whitney U statistic.

    Positive values mean larger values are more
    associated with Target=1; negative values mean
    larger values are more associated with Target=0.
    """
    return (2 * u) / (n0 * n1) - 1


def main():

    df = pd.read_csv(DATA_PATH)

    print("=" * 90)
    print("NETWORK SOCIAL-DYNAMICS ANALYSIS — TRAINING DATA")
    print("=" * 90)

    results = []

    for feature in NETWORK_FEATURES:

        group0 = df.loc[
            df["Target"] == 0,
            feature
        ]

        group1 = df.loc[
            df["Target"] == 1,
            feature
        ]

        # U is computed with emergent cascades (Target=1)
        # as the first sample so effect direction is intuitive.
        u_stat, p_value = mannwhitneyu(
            group1,
            group0,
            alternative="two-sided",
        )

        effect = rank_biserial_from_u(
            u_stat,
            len(group1),
            len(group0),
        )

        results.append(
            {
                "Feature": feature,

                "NonEmergent_Mean":
                    group0.mean(),

                "Emergent_Mean":
                    group1.mean(),

                "NonEmergent_Median":
                    group0.median(),

                "Emergent_Median":
                    group1.median(),

                "U_Statistic":
                    u_stat,

                "P_Value":
                    p_value,

                "RankBiserial":
                    effect,
            }
        )

    results_df = pd.DataFrame(results)

    reject, adjusted_p, _, _ = multipletests(
    results_df["P_Value"],
    alpha=0.05,
    method="fdr_bh",
)

    results_df["Adjusted_P"] = adjusted_p
    results_df["Significant_FDR"] = reject

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 90)
    print("TRAINING FEATURE CORRELATIONS")
    print("=" * 90)

    print(
        df[NETWORK_FEATURES]
        .corr(method="spearman")
        .round(3)
    )

    


if __name__ == "__main__":
    main()