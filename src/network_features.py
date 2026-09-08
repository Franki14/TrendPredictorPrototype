from pathlib import Path
from collections import Counter
import math
import pickle

import networkx as nx
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "twitter"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Constants
# ============================================================

OBSERVATION_SIZE = 5


# ============================================================
# File mappings
# ============================================================

RAW_SPLIT_FILES = {
    "train": "cascadetrain.txt",
    "validation": "cascadevalid.txt",
    "test": "cascadetest.txt",
}

PROCESSED_FILES = {
    "train": "cascade_train.csv",
    "validation": "cascade_validation.csv",
    "test": "cascade_test.csv",
}

OUTPUT_FILES = {
    "train": "cascade_train_network.csv",
    "validation": "cascade_validation_network.csv",
    "test": "cascade_test_network.csv",
}


# ============================================================
# Load user-index mapping
# ============================================================

def load_user_mapping():
    """
    Load the mapping from original Twitter user IDs to the
    integer user indices used in the cascade split files.
    """

    mapping_path = RAW_DIR / "u2idx.pickle"

    with open(mapping_path, "rb") as f:
        u2idx = pickle.load(f)

    return u2idx


# ============================================================
# Build social graph
# ============================================================

def build_social_graph(u2idx):
    """
    Build an undirected graph from edges.txt.

    The raw edge file contains reciprocal directed records,
    so nx.Graph collapses each reciprocal pair into one
    undirected relationship.

    Original Twitter user IDs are mapped to the integer indices
    used in the cascade split files.
    """

    graph = nx.Graph()

    edges_path = RAW_DIR / "edges.txt"

    skipped_records = 0
    mapped_records = 0

    with open(edges_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Handle either comma-separated or whitespace-separated edges.
            if "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            if len(parts) != 2:
                skipped_records += 1
                continue

            source_original = parts[0].strip()
            target_original = parts[1].strip()

            if (
                source_original not in u2idx
                or target_original not in u2idx
            ):
                skipped_records += 1
                continue

            source = u2idx[source_original]
            target = u2idx[target_original]

            # Ignore special mapping tokens such as <blank> and </s>.
            if source < 2 or target < 2:
                continue

            # Ignore self-loops.
            if source == target:
                continue

            graph.add_edge(
                source,
                target,
            )

            mapped_records += 1

    print(
        f"Graph loaded: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print(
        f"Mapped edge records: "
        f"{mapped_records}"
    )

    print(
        f"Skipped edge records: "
        f"{skipped_records}"
    )

    return graph


# ============================================================
# Load first five adopters
# ============================================================

def load_first_five(filename):
    """
    Extract the first five chronologically ordered adopters
    from every eligible cascade.

    CascadeID preserves the original zero-based line number
    from the corresponding raw split file so that it matches
    cascade_preprocess.py.
    """

    cascades = []

    file_path = RAW_DIR / filename

    with open(file_path, "r") as f:

        for cascade_id, line in enumerate(f):

            events = []

            for event in line.strip().split():

                try:
                    user, timestamp = event.split(",")

                    events.append(
                        (
                            int(user),
                            float(timestamp),
                        )
                    )

                except ValueError:
                    continue

            events.sort(
                key=lambda x: x[1]
            )

            if len(events) < OBSERVATION_SIZE:
                continue

            early_users = [
                user
                for user, _ in events[:OBSERVATION_SIZE]
            ]

            cascades.append(
                {
                    "CascadeID": cascade_id,
                    "EarlyUsers": early_users,
                }
            )

    return cascades


# ============================================================
# Community diversity
# ============================================================

def calculate_community_diversity(
    early_users,
    community_lookup,
):
    """
    Calculate normalized Shannon entropy of community
    membership among the first five adopters.

    Interpretation:

        0.0 = all five adopters are in the same community
        1.0 = all five adopters are in different communities
    """

    community_ids = [
        community_lookup[user]
        for user in early_users
    ]

    counts = Counter(
        community_ids
    )

    total = len(
        community_ids
    )

    entropy = 0.0

    for count in counts.values():

        probability = (
            count / total
        )

        entropy -= (
            probability
            * math.log(probability)
        )

    max_entropy = math.log(
        total
    )

    if max_entropy == 0:
        return 0.0

    return (
        entropy / max_entropy
    )


# ============================================================
# Calculate network features
# ============================================================

def calculate_network_features(
    cascades,
    graph,
    clustering_coefficients,
    community_lookup,
    pagerank_scores,
):
    """
    Calculate network features using only the first five
    adopters of each cascade.
    """

    rows = []

    for cascade in cascades:

        cascade_id = (
            cascade["CascadeID"]
        )

        early_users = (
            cascade["EarlyUsers"]
        )

        # ----------------------------------------------------
        # Validate users
        # ----------------------------------------------------

        missing_users = [
            user
            for user in early_users
            if user not in graph
        ]

        if missing_users:

            raise ValueError(
                f"Cascade {cascade_id} contains users "
                f"not present in graph: {missing_users}"
            )

        # ----------------------------------------------------
        # Degree features
        # ----------------------------------------------------

        degrees = [
            graph.degree(user)
            for user in early_users
        ]

        mean_degree = (
            sum(degrees)
            / len(degrees)
        )

        max_degree = max(
            degrees
        )

        degree_series = pd.Series(
            degrees,
            dtype=float,
        )

        degree_std = (
            degree_series.std(
                ddof=0
            )
        )

        # ----------------------------------------------------
        # Internal connectivity
        # ----------------------------------------------------

        early_subgraph = (
            graph.subgraph(
                early_users
            )
        )

        early_internal_edges = (
            early_subgraph
            .number_of_edges()
        )

        possible_edges = (
            OBSERVATION_SIZE
            * (OBSERVATION_SIZE - 1)
            / 2
        )

        early_density = (
            early_internal_edges
            / possible_edges
        )

        # ----------------------------------------------------
        # Neighbourhood reach
        # ----------------------------------------------------

        neighbours = set()

        for user in early_users:

            neighbours.update(
                graph.neighbors(user)
            )

        neighbours.difference_update(
            early_users
        )

        neighbourhood_reach = (
            len(neighbours)
        )

        # ----------------------------------------------------
        # Local clustering
        # ----------------------------------------------------

        clustering_values = [
            clustering_coefficients[user]
            for user in early_users
        ]

        mean_clustering = (
            sum(clustering_values)
            / len(clustering_values)
        )

        # ----------------------------------------------------
        # Community features
        # ----------------------------------------------------

        early_communities = [
            community_lookup[user]
            for user in early_users
        ]

        early_community_count = (
            len(
                set(early_communities)
            )
        )

        community_diversity = (
            calculate_community_diversity(
                early_users,
                community_lookup,
            )
        )

        # ----------------------------------------------------
        # PageRank features
        # ----------------------------------------------------

        pagerank_values = [
            pagerank_scores[user]
            for user in early_users
        ]

        mean_pagerank = (
            sum(pagerank_values)
            / len(pagerank_values)
        )

        max_pagerank = max(
            pagerank_values
        )

        # ----------------------------------------------------
        # Store feature row
        # ----------------------------------------------------

        rows.append(
            {
                "CascadeID":
                    cascade_id,

                "MeanDegree":
                    mean_degree,

                "MaxDegree":
                    max_degree,

                "DegreeStd":
                    degree_std,

                "EarlyInternalEdges":
                    early_internal_edges,

                "EarlyDensity":
                    early_density,

                "NeighbourhoodReach":
                    neighbourhood_reach,

                "MeanClustering":
                    mean_clustering,

                "EarlyCommunityCount":
                    early_community_count,

                "CommunityDiversity":
                    community_diversity,

                "MeanPageRank":
                    mean_pagerank,

                "MaxPageRank":
                    max_pagerank,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Process one dataset split
# ============================================================

def process_split(
    split_name,
    graph,
    clustering_coefficients,
    community_lookup,
    pagerank_scores,
):
    """
    Generate network features for one split and merge them with
    the existing temporal feature dataset.
    """

    print()
    print("=" * 60)
    print(
        split_name.upper()
    )
    print("=" * 60)

    raw_filename = (
        RAW_SPLIT_FILES[
            split_name
        ]
    )

    processed_filename = (
        PROCESSED_FILES[
            split_name
        ]
    )

    output_filename = (
        OUTPUT_FILES[
            split_name
        ]
    )

    # --------------------------------------------------------
    # Load first-five adopter sequences
    # --------------------------------------------------------

    cascades = (
        load_first_five(
            raw_filename
        )
    )

    network_df = (
        calculate_network_features(
            cascades,
            graph,
            clustering_coefficients,
            community_lookup,
            pagerank_scores,
        )
    )

    # --------------------------------------------------------
    # Load existing temporal dataset
    # --------------------------------------------------------

    processed_path = (
        PROCESSED_DIR
        / processed_filename
    )

    processed_df = (
        pd.read_csv(
            processed_path
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_df = (
        processed_df.merge(
            network_df,
            on="CascadeID",
            how="left",
            validate="one_to_one",
        )
    )

    network_columns = [
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

    # --------------------------------------------------------
    # Diagnose missing matches
    # --------------------------------------------------------

    missing = merged_df[
        merged_df[
            network_columns
        ]
        .isna()
        .any(axis=1)
    ]

    if not missing.empty:

        print(
            "\nMissing network-feature matches:"
        )

        print(
            missing[
                [
                    "CascadeID",
                    "FinalSize",
                    "Target",
                ]
            ]
            .head(20)
        )

        print(
            "Number of unmatched cascades:",
            len(missing),
        )

        raise ValueError(
            f"{split_name}: "
            f"network features missing for "
            f"{len(missing)} cascades."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert (
        merged_df[
            network_columns
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )

    assert (
        len(merged_df)
        == len(processed_df)
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        f"Shape: "
        f"{merged_df.shape}"
    )

    print(
        "\nNetwork feature statistics:"
    )

    print(
        merged_df[
            network_columns
        ]
        .describe()
        .round(6)
    )

    print(
        "\nMean features by target:"
    )

    print(
        merged_df.groupby(
            "Target"
        )[
            network_columns
        ]
        .mean()
        .round(6)
    )

    print(
        "\nPageRank features by target:"
    )

    print(
        merged_df.groupby(
            "Target"
        )[
            [
                "MeanPageRank",
                "MaxPageRank",
            ]
        ]
        .mean()
        .round(8)
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR
        / output_filename
    )

    merged_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: "
        f"{output_path}"
    )

    return merged_df


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    print(
        "Building social graph..."
    )

    u2idx = (
        load_user_mapping()
    )

    graph = (
        build_social_graph(
            u2idx
        )
    )

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    print(
        "\nComputing PageRank..."
    )

    pagerank_scores = (
        nx.pagerank(
            graph,
            alpha=0.85,
        )
    )

    print(
        "PageRank complete."
    )

    # --------------------------------------------------------
    # Clustering coefficients
    # --------------------------------------------------------

    print(
        "\nComputing node clustering coefficients..."
    )

    clustering_coefficients = (
        nx.clustering(
            graph
        )
    )

    print(
        "Clustering coefficients complete."
    )

    # --------------------------------------------------------
    # Louvain communities
    # --------------------------------------------------------

    print(
        "\nDetecting Louvain communities..."
    )

    communities = (
        nx.community.louvain_communities(
            graph,
            seed=42,
        )
    )

    community_lookup = {}

    for community_id, nodes in enumerate(
        communities
    ):

        for node in nodes:

            community_lookup[
                node
            ] = community_id

    print(
        f"Louvain communities detected: "
        f"{len(communities)}"
    )

    assert (
        len(community_lookup)
        == graph.number_of_nodes()
    )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        process_split(
            split_name,
            graph,
            clustering_coefficients,
            community_lookup,
            pagerank_scores,
        )

    print()
    print(
        "Network feature engineering complete."
    )


if __name__ == "__main__":
    main()