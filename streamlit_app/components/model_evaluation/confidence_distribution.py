import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


def prepare_confidence_data(data):
    """Prepare confidence distribution data"""
    conf_data = []

    for model, files in data.items():
        if "sklearn_predictions" in files:
            df = files["sklearn_predictions"]
            kde = gaussian_kde(df["confidence_positive"])
            x_range = np.linspace(0, 1, 200)
            y_values = kde(x_range)

            conf_data.append(
                {
                    "model": model,
                    "x_range": x_range,
                    "y_values": y_values,
                    "color_idx": list(data.keys()).index(model),
                }
            )

    return conf_data


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_confidence_plot(conf_data):
    """Create interactive Plotly confidence distribution plot"""
    fig = go.Figure()

    for data in conf_data:
        fig.add_trace(
            go.Scatter(
                x=data["x_range"],
                y=data["y_values"],
                name=data["model"],
                mode="lines",
                fill="tozeroy",
                fillcolor=px.colors.qualitative.Set3[
                    data["color_idx"] % len(px.colors.qualitative.Set3)
                ],
                line=dict(width=2),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Density",
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(range=[0, 1], tickformat=".1f"),
        template="plotly_white",
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_confidence_plot(conf_data):
    """Create static Matplotlib confidence distribution plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(conf_data)))

    for data, color in zip(conf_data, colors):
        ax.fill_between(
            data["x_range"],
            data["y_values"],
            alpha=0.3,
            color=color,
            label=data["model"],
        )
        ax.plot(data["x_range"], data["y_values"], color=color, linewidth=2)

    # Customize plot
    ax.set_title(
        "Prediction Confidence Distribution", pad=20, fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Set x-axis range and format
    ax.set_xlim(0, 1)

    # Add grid and remove spines
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    plt.tight_layout()
    return fig


def confidence_distribution_component(data):
    """Component for displaying prediction confidence distributions using KDE curves"""

    st.markdown(
        """
        Analyze the distribution of model prediction confidences.
        This helps understand how certain the model is about its predictions.
    """
    )

    # Prepare data
    conf_data = prepare_confidence_data(data)

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_confidence_plot(conf_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_confidence_plot(conf_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Confidence Distribution"):
        st.markdown(
            """
            - Shows how confident the model is in its predictions
            - Smooth curves represent the density of predictions at each confidence level
            - Peaks indicate common confidence values
            - Ideal distribution often shows clear separation between classes
            - Compare patterns across different models:
                - Sharp peaks suggest high certainty
                - Broad distributions suggest uncertainty
                - Multiple peaks may indicate distinct prediction patterns
        """
        )
