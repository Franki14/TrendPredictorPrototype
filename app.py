"""
TrendPredictor Dashboard
========================

Interactive research prototype for exploring early social-media
trend-emergence predictions produced by the final Balanced
Random Forest model.

Dashboard sections:
1. Trend Predictions
2. Trend Explorer
3. Feature Analysis

The dashboard uses the frozen model and existing experimental
outputs. It does not retrain or modify the predictive model.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from dashboard_data import (
    load_dashboard_data,
    build_dashboard_record,
)

from predict import (
    load_model,
    load_metadata,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="TrendPredictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Dashboard Styling
# ============================================================

def load_css():
    """
    Load the external dashboard stylesheet if it exists.
    """

    css_path = PROJECT_ROOT / "assets" / "style.css"

    if not css_path.exists():
        return

    with open(
        css_path,
        "r",
        encoding="utf-8",
    ) as file:

        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True,
        )


load_css()


# ============================================================
# Friendly Feature Labels
# ============================================================

FEATURE_LABELS = {
    # Temporal
    "TimeTo5": "Time to First 5 Interactions",
    "MeanInterarrival": "Mean Time Between Interactions",
    "StdInterarrival": "Variation in Interaction Timing",
    "MinInterarrival": "Minimum Time Between Interactions",
    "MaxInterarrival": "Maximum Time Between Interactions",
    "EarlyVelocity": "Early Growth Velocity",
    "EarlyAcceleration": "Early Growth Acceleration",

    # Network
    "MeanDegree": "Mean Network Degree",
    "MaxDegree": "Maximum Network Degree",
    "DegreeStd": "Network Degree Variation",
    "EarlyInternalEdges": "Early Internal Connections",
    "EarlyDensity": "Early Network Density",
    "NeighbourhoodReach": "Network Neighbourhood Reach",
    "MeanClustering": "Mean Clustering Coefficient",
    "EarlyCommunityCount": "Early Community Count",
    "CommunityDiversity": "Community Diversity",
    "MeanPageRank": "Mean PageRank",
    "MaxPageRank": "Maximum PageRank",
}


FEATURE_DESCRIPTIONS = {
    "TimeTo5":
        "Time required for the cascade to reach its first five "
        "observed interactions.",

    "MinInterarrival":
        "Shortest observed interval between early interactions.",

    "EarlyAcceleration":
        "Change in the rate of activity during the early cascade.",

    "NeighbourhoodReach":
        "Extent of the surrounding network reached during the "
        "early cascade.",

    "EarlyDensity":
        "Connectivity among nodes involved in the early cascade.",

    "MeanClustering":
        "Average local clustering of nodes involved in the cascade.",

    "CommunityDiversity":
        "Extent to which the early cascade spans different "
        "network communities.",

    "MeanPageRank":
        "Average PageRank centrality of nodes involved in the "
        "early cascade.",
}


# ============================================================
# Cached Resources
# ============================================================

@st.cache_resource
def get_model():
    """
    Load the frozen Random Forest model once per application
    session.
    """

    return load_model()


@st.cache_resource
def get_metadata():
    """
    Load final model metadata.
    """

    return load_metadata()


@st.cache_data
def get_dashboard_data():
    """
    Load historical cascade records used by the prototype.
    """

    return load_dashboard_data()


# ============================================================
# Load Model and Data
# ============================================================

try:

    model = get_model()
    metadata = get_metadata()
    dashboard_df = get_dashboard_data()

except FileNotFoundError as error:

    st.error(
        "The trained model or another required application "
        "resource could not be loaded."
    )

    st.write(
        "Generate the frozen model before launching the dashboard:"
    )

    st.code(
        "python src/train_final_model.py"
    )

    with st.expander("Technical details"):
        st.exception(error)

    st.stop()

except Exception as error:

    st.error(
        "TrendPredictor could not initialise correctly."
    )

    with st.expander("Technical details"):
        st.exception(error)

    st.stop()


# ============================================================
# Application Header
# ============================================================

st.markdown(
    """
    <div class="tp-header">
        <h1>TrendPredictor</h1>
        <p class="tp-subtitle">
            Early Social Media Trend Emergence Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Explore early cascade behaviour, model predictions and "
    "the factors associated with trend emergence."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("TrendPredictor")

st.sidebar.caption(
    "Research Prototype"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Trend Predictions",
        "Trend Explorer",
        "Feature Analysis",
    ],
)

st.sidebar.divider()

st.sidebar.subheader(
    "Cascade Selection"
)

cascade_ids = (
    dashboard_df["CascadeID"]
    .sort_values()
    .tolist()
)

selected_cascade = st.sidebar.number_input(
    "Cascade ID",
    min_value=int(min(cascade_ids)),
    max_value=int(max(cascade_ids)),
    value=int(cascade_ids[0]),
    step=1,
)

st.sidebar.caption(
    f"{len(cascade_ids)} historical cascades available "
    f"(ID range {min(cascade_ids)}–{max(cascade_ids)})."
)

if selected_cascade not in cascade_ids:

    st.sidebar.error(
        f"Cascade {selected_cascade} is not available."
    )

    st.info(
        "Please enter an available Cascade ID using the sidebar."
    )

    st.stop()

st.sidebar.divider()

st.sidebar.subheader(
    "Model"
)

st.sidebar.write(
    "**Balanced Random Forest**"
)

st.sidebar.caption(
    "8 predictive features"
)

st.sidebar.caption(
    "500 trees"
)

st.sidebar.caption(
    "Classification threshold: 0.50"
)


# ============================================================
# Build Selected Dashboard Record
# ============================================================

try:

    record = build_dashboard_record(
        selected_cascade,
        df=dashboard_df,
        model=model,
        metadata=metadata,
    )

except Exception as error:

    st.error(
        "The selected cascade could not be processed."
    )

    with st.expander("Technical details"):
        st.exception(error)

    st.stop()


historical = record["historical"]
prediction = record["prediction"]
characteristics = record["characteristics"]
model_features = record["model_features"]


# ============================================================
# Helper Functions
# ============================================================

def friendly_feature_name(feature):
    """
    Convert an internal feature name into a user-facing label.
    """

    return FEATURE_LABELS.get(
        feature,
        feature,
    )


def display_figure(
    filepath,
    missing_message,
):
    """
    Display an existing experimental figure safely.
    """

    if filepath.exists():

        st.image(
            str(filepath),
            width="stretch",
        )

    else:

        st.warning(
            missing_message
        )


def format_value(value):
    """
    Format numerical characteristics for dashboard display.
    """

    if value is None:
        return "Not available"

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:.3f}"

    return f"{value:.6f}"


# ============================================================
# PAGE 1 — Trend Predictions
# ============================================================

if page == "Trend Predictions":

    st.header(
        "Trend Predictions"
    )

    st.write(
        "Review the final model's early trend-emergence "
        "prediction for the selected historical cascade."
    )

    st.info(
        "The Viral Potential Score (VPS) is the model's "
        "positive-class probability expressed on a 0–100 "
        "scale. It represents model confidence and should not "
        "be interpreted as a guarantee that a topic will "
        "become viral."
    )

    # --------------------------------------------------------
    # Selected Cascade
    # --------------------------------------------------------

    st.subheader(
        f"Cascade {historical['cascade_id']}"
    )

    st.caption(
        f"Historical dataset split: {historical['split']}"
    )

    # --------------------------------------------------------
    # Main Prediction Cards
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
<div class="tp-card">
    <div class="tp-card-label">VIRAL POTENTIAL SCORE</div>
    <div class="tp-card-value">{prediction["vps"]:.1f}/100</div>
    <div class="tp-card-caption">VPS level: {prediction["vps_category"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
<div class="tp-card">
    <div class="tp-card-label">PREDICTED OUTCOME</div>
    <div class="tp-card-value">{prediction["prediction_label"]}</div>
    <div class="tp-card-caption">Threshold: {prediction["classification_threshold"]:.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
<div class="tp-card">
    <div class="tp-card-label">MODEL PROBABILITY</div>
    <div class="tp-card-value">{prediction["probability_percent"]:.1f}%</div>
    <div class="tp-card-caption">Positive emergence probability</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.subheader(
        "Prediction Confidence"
    )

    st.progress(
        float(
            prediction["probability"]
        )
    )

    st.write(
        f"The model assigned a "
        f"**{prediction['probability_percent']:.1f}%** "
        "probability to the positive trend-emergence class."
    )

    if prediction["probability"] >= prediction[
        "classification_threshold"
    ]:

        st.write(
            "Because this value is at or above the "
            "classification threshold, the cascade is "
            "classified as **Emerging Trend**."
        )

    else:

        st.write(
            "Because this value is below the classification "
            "threshold, the cascade is classified as "
            "**Non-Emerging Trend**."
        )

    st.divider()

    # --------------------------------------------------------
    # Historical Outcome
    # --------------------------------------------------------

    st.subheader(
        "Historical Outcome"
    )

    hist1, hist2 = st.columns(2)

    with hist1:

        st.metric(
            "Actual Outcome",
            historical["actual_label"],
        )

    with hist2:

        st.metric(
            "Final Cascade Size",
            historical["final_size"],
        )

    if record["prediction_correct"]:

        status_text = (
            "MATCH — prediction agrees with the historical outcome"
        )

    else:

        status_text = (
            "MISMATCH — prediction differs from the historical outcome"
        )

        st.markdown(
        f"""
    <span class="tp-status">{status_text}</span>
    """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The historical target and final cascade size are shown "
        "only for retrospective evaluation. Neither value is "
        "supplied to the predictive model."
    )

    st.divider()

    # --------------------------------------------------------
    # Model Inputs
    # --------------------------------------------------------

    st.subheader(
        "Prediction Inputs"
    )

    st.write(
        "The final model generated this prediction from eight "
        "early temporal and network characteristics."
    )

    feature_rows = []

    for feature, value in model_features.items():

        feature_rows.append(
            {
                "Feature": friendly_feature_name(
                    feature
                ),
                "Internal Name": feature,
                "Value": value,
                "Feature Group": (
                    "Temporal"
                    if feature in [
                        "TimeTo5",
                        "MinInterarrival",
                        "EarlyAcceleration",
                    ]
                    else "Network"
                ),
            }
        )

    feature_table = pd.DataFrame(
        feature_rows
    )

    st.dataframe(
        feature_table,
        hide_index=True,
        width="stretch",
    )

    with st.expander(
        "What do these features mean?"
    ):

        for feature in metadata["features"]:

            st.markdown(
                f"**{friendly_feature_name(feature)}**"
            )

            st.write(
                FEATURE_DESCRIPTIONS.get(
                    feature,
                    "No description available.",
                )
            )


# ============================================================
# PAGE 2 — Trend Explorer
# ============================================================

elif page == "Trend Explorer":

    st.header(
        "Trend Explorer"
    )

    st.write(
        "Explore the temporal and network characteristics of "
        "the selected historical cascade."
    )

    st.subheader(
        f"Cascade {historical['cascade_id']}"
    )

    summary1, summary2, summary3 = st.columns(3)

    with summary1:

        st.metric(
            "Historical Final Size",
            historical["final_size"],
        )

    with summary2:

        st.metric(
            "Historical Outcome",
            historical["actual_label"],
        )

    with summary3:

        st.metric(
            "VPS",
            f"{prediction['vps']:.1f}/100",
        )

    st.caption(
        "Final size and historical outcome are retrospective "
        "information and are not model inputs."
    )

    st.divider()

    # --------------------------------------------------------
    # Temporal Characteristics
    # --------------------------------------------------------

    st.subheader(
        "Temporal Characteristics"
    )

    st.write(
        "These measurements describe the timing and early "
        "growth behaviour of the cascade."
    )

    temporal_feature_names = [
        "TimeTo5",
        "MeanInterarrival",
        "StdInterarrival",
        "MinInterarrival",
        "MaxInterarrival",
        "EarlyVelocity",
        "EarlyAcceleration",
    ]

    temporal_rows = []

    for feature in temporal_feature_names:

        if feature in characteristics:

            temporal_rows.append(
                {
                    "Characteristic":
                        friendly_feature_name(
                            feature
                        ),

                    "Internal Name":
                        feature,

                    "Value":
                        format_value(
                            characteristics[
                                feature
                            ]
                        ),

                    "Used by Final Model":
                        (
                            "Yes"
                            if feature in metadata[
                                "features"
                            ]
                            else "No"
                        ),
                }
            )

    temporal_df = pd.DataFrame(
        temporal_rows
    )

    st.dataframe(
        temporal_df,
        hide_index=True,
        width="stretch",
    )

    st.divider()

    # --------------------------------------------------------
    # Network Characteristics
    # --------------------------------------------------------

    st.subheader(
        "Network Characteristics"
    )

    st.write(
        "These measurements describe the structure and reach "
        "of the network involved in the early cascade."
    )

    network_feature_names = [
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

    network_rows = []

    for feature in network_feature_names:

        if feature in characteristics:

            network_rows.append(
                {
                    "Characteristic":
                        friendly_feature_name(
                            feature
                        ),

                    "Internal Name":
                        feature,

                    "Value":
                        format_value(
                            characteristics[
                                feature
                            ]
                        ),

                    "Used by Final Model":
                        (
                            "Yes"
                            if feature in metadata[
                                "features"
                            ]
                            else "No"
                        ),
                }
            )

    network_df = pd.DataFrame(
        network_rows
    )

    st.dataframe(
        network_df,
        hide_index=True,
        width="stretch",
    )

    st.info(
        "The Trend Explorer contains more characteristics than "
        "the final predictive model. The 'Used by Final Model' "
        "column identifies the eight variables that actually "
        "contribute to prediction."
    )


# ============================================================
# PAGE 3 — Feature Analysis
# ============================================================

elif page == "Feature Analysis":

    st.header(
        "Feature Analysis"
    )

    st.write(
        "Explore the factors used by the final Balanced Random "
        "Forest and the experimental evidence used to interpret "
        "its predictions."
    )

    st.info(
        "The analyses on this page are global model analyses "
        "performed on the validation data. They explain overall "
        "model behaviour rather than the prediction for one "
        "specific cascade."
    )

    # --------------------------------------------------------
    # Final Predictive Features
    # --------------------------------------------------------

    st.subheader(
        "Final Predictive Features"
    )

    st.write(
        "The final model combines three temporal features and "
        "five network-derived features."
    )

    temporal_model_features = [
        "TimeTo5",
        "MinInterarrival",
        "EarlyAcceleration",
    ]

    network_model_features = [
        "NeighbourhoodReach",
        "EarlyDensity",
        "MeanClustering",
        "CommunityDiversity",
        "MeanPageRank",
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Temporal Features"
        )

        for feature in temporal_model_features:

            st.markdown(
                f"**{friendly_feature_name(feature)}**"
            )

            st.caption(
                FEATURE_DESCRIPTIONS[
                    feature
                ]
            )

    with col2:

        st.markdown(
            "#### Network Features"
        )

        for feature in network_model_features:

            st.markdown(
                f"**{friendly_feature_name(feature)}**"
            )

            st.caption(
                FEATURE_DESCRIPTIONS[
                    feature
                ]
            )

    st.divider()

    # --------------------------------------------------------
    # Permutation Importance
    # --------------------------------------------------------

    st.subheader(
        "Permutation Feature Importance"
    )

    st.write(
        "Permutation importance measures the reduction in "
        "validation ROC-AUC after a feature is randomly "
        "shuffled. Larger reductions indicate that the model "
        "relies more strongly on that feature for "
        "discrimination."
    )

    permutation_figure = (
        FIGURE_DIR /
        "permutation_importance.png"
    )

    display_figure(
        permutation_figure,
        "Permutation importance figure not found.",
    )

    st.markdown(
        """
        **Interpretation:** Network Neighbourhood Reach and
        Early Network Density produced the largest reductions
        in validation ROC-AUC when permuted. Early Growth
        Acceleration and Minimum Time Between Interactions also
        provided useful predictive information.
        """
    )

    with st.expander(
        "View permutation importance values"
    ):

        permutation_df = pd.DataFrame(
            {
                "Feature": [
                    "Network Neighbourhood Reach",
                    "Early Network Density",
                    "Early Growth Acceleration",
                    "Minimum Time Between Interactions",
                    "Mean Clustering Coefficient",
                    "Mean PageRank",
                    "Time to First 5 Interactions",
                    "Community Diversity",
                ],

                "Mean ROC-AUC Decrease": [
                    0.1254,
                    0.1214,
                    0.0507,
                    0.0467,
                    0.0346,
                    0.0234,
                    0.0207,
                    0.0148,
                ],

                "Standard Deviation": [
                    0.0205,
                    0.0288,
                    0.0132,
                    0.0137,
                    0.0124,
                    0.0119,
                    0.0114,
                    0.0048,
                ],
            }
        )

        st.dataframe(
            permutation_df,
            hide_index=True,
            width="stretch",
        )

    st.divider()

    # --------------------------------------------------------
    # Partial Dependence
    # --------------------------------------------------------

    st.subheader(
        "Partial Dependence"
    )

    st.write(
        "Partial dependence examines how the model's predicted "
        "positive-class response changes across selected "
        "feature values while averaging over the other "
        "predictors."
    )

    pdp_figure = (
        FIGURE_DIR /
        "partial_dependence.png"
    )

    display_figure(
        pdp_figure,
        "Partial-dependence figure not found.",
    )

    st.markdown(
        """
        **Interpretation:** The model learned several
        non-linear relationships. For example, Early Growth
        Acceleration shows a substantial change in the model
        response across its observed range, while the network
        variables also exhibit threshold-like behaviour.
        These patterns are consistent with the use of a Random
        Forest, which can capture non-linear relationships
        without requiring a predefined linear form.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Network Interaction
    # --------------------------------------------------------

    st.subheader(
        "Network Feature Interaction"
    )

    st.write(
        "A two-way partial-dependence analysis examines the "
        "joint model response across Network Neighbourhood "
        "Reach and Early Network Density."
    )

    interaction_figure = (
        FIGURE_DIR /
        "network_interaction_pdp.png"
    )

    display_figure(
        interaction_figure,
        "Network interaction figure not found.",
    )

    st.markdown(
        """
        **Interpretation:** The predicted response changes
        across combinations of network reach and density rather
        than varying with either feature independently. This
        demonstrates an interaction pattern represented by the
        Random Forest.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Feature Group Ablation
    # --------------------------------------------------------

    st.subheader(
        "Temporal vs Network Features"
    )

    st.write(
        "The feature-group ablation experiment compared "
        "temporal features alone, network features alone, and "
        "the combined eight-feature representation."
    )

    ablation_figure = (
        FIGURE_DIR /
        "feature_group_ablation.png"
    )

    display_figure(
        ablation_figure,
        "Feature-group ablation figure not found.",
    )

    ablation_df = pd.DataFrame(
        {
            "Feature Group": [
                "Temporal Only",
                "Network Only",
                "Combined",
            ],

            "Number of Features": [
                3,
                5,
                8,
            ],

            "F1": [
                0.3679,
                0.5714,
                0.6869,
            ],

            "ROC-AUC": [
                0.5402,
                0.7758,
                0.8584,
            ],

            "PR-AUC": [
                0.3377,
                0.5909,
                0.7287,
            ],
        }
    )

    st.dataframe(
        ablation_df,
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        """
        **Key finding:** The combined temporal and network
        representation achieved higher validation F1,
        ROC-AUC and PR-AUC than either feature group used
        alone. This supports the project's use of an integrated
        feature representation.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Interpretation Warning
    # --------------------------------------------------------

    st.warning(
        "Model interpretation is not causal inference. Feature "
        "importance and partial dependence describe patterns "
        "learned by the predictive model and do not demonstrate "
        "that changing a feature would cause a social-media "
        "trend to emerge."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "TrendPredictor — Predictive Modelling of Social Media "
    "Trend Emergence"
)

st.caption(
    "Research prototype. Predictions represent statistical "
    "model outputs and should not be interpreted as guarantees "
    "of future social-media behaviour."
)