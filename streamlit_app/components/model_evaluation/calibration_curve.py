import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def prepare_calibration_data(data):
    """Prepare calibration data"""
    calibration_data = []

    for model, files in data.items():
        if "sklearn_predictions" in files:
            df = files["sklearn_predictions"]
            prob_true, prob_pred = calibration_curve(
                df["true_label"], df["confidence_positive"], n_bins=10
            )

            calibration_data.append(
                {"model": model, "prob_true": prob_true, "prob_pred": prob_pred}
            )

    return calibration_data


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_calibration_plot(calibration_data):
    """Create interactive Plotly calibration plot"""
    fig = go.Figure()

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        )
    )

    for data in calibration_data:
        fig.add_trace(
            go.Scatter(
                x=data["prob_pred"],
                y=data["prob_true"],
                name=data["model"],
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_calibration_plot(calibration_data):
    """Create static Matplotlib calibration plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    # Plot calibration curves
    colors = plt.cm.Set3(np.linspace(0, 1, len(calibration_data)))

    for data, color in zip(calibration_data, colors):
        ax.plot(
            data["prob_pred"],
            data["prob_true"],
            marker="o",
            color=color,
            label=data["model"],
        )

    # Customize plot
    ax.set_title("Calibration Plot", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add grid and remove spines
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    plt.tight_layout()
    return fig


def calibration_component(data):
    """Component for model calibration analysis"""

    st.markdown(
        """
        Analyze model calibration to understand how well predicted probabilities
        match actual outcomes.
    """
    )

    # Prepare data
    calibration_data = prepare_calibration_data(data)

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_calibration_plot(calibration_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_calibration_plot(calibration_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Calibration"):
        st.markdown(
            """
            - Perfect calibration follows diagonal line
            - Above line: Model is underconfident
            - Below line: Model is overconfident
            - Well-calibrated models have:
                - Reliable probability estimates
                - Better uncertainty quantification
            - Important for:
                - Decision making
                - Risk assessment
                - Model reliability
        """
        )
