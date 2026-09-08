from pathlib import Path
import pickle
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "twitter"


def load_user_mapping():
    """
    Load the mapping from original Twitter IDs to
    processed cascade indices.
    """
    with open(RAW_DIR / "u2idx.pickle", "rb") as f:
        return pickle.load(f)


def build_social_graph():
    """
    Build an undirected social graph.

    Edge direction is not interpreted because the available
    repository code does not document follower/followee
    orientation. Each pair is therefore treated conservatively
    as a social-network connection.
    """

    u2idx = load_user_mapping()

    graph = nx.Graph()

    skipped = 0

    with open(RAW_DIR / "edges.txt", "r") as f:

        for line in f:

            try:
                source_original, target_original = (
                    line.strip().split(",")
                )

                source = u2idx.get(source_original)
                target = u2idx.get(target_original)

                if source is None or target is None:
                    skipped += 1
                    continue

                # Ignore self-loops for structural analysis.
                if source == target:
                    continue

                graph.add_edge(source, target)

            except ValueError:
                skipped += 1

    return graph, skipped


def main():

    print("Loading social network...")

    graph, skipped = build_social_graph()

    print("\n" + "=" * 60)
    print("SOCIAL NETWORK SUMMARY")
    print("=" * 60)

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Skipped records:", skipped)

    print(
        "Density:",
        round(nx.density(graph), 6)
    )

    degrees = [
        degree
        for _, degree in graph.degree()
    ]

    print(
        "Average degree:",
        round(
            sum(degrees) / len(degrees),
            2
        )
    )

    print(
        "Minimum degree:",
        min(degrees)
    )

    print(
        "Maximum degree:",
        max(degrees)
    )

    components = list(
        nx.connected_components(graph)
    )

    print(
        "Connected components:",
        len(components)
    )

    largest_component = max(
        components,
        key=len
    )

    print(
        "Largest component:",
        len(largest_component)
    )

    print(
        "Largest-component proportion:",
        f"{len(largest_component) / graph.number_of_nodes():.2%}"
    )


if __name__ == "__main__":
    main()