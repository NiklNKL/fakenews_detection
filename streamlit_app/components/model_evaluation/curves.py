import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def prepare_curves_data(data):
    """Prepare ROC and PR curve data"""
    curves_data = []

    for model, files in data.items():
        model_data = {"model": model}

        if "sklearn_roc" in files:
            roc_df = files["sklearn_roc"]
            model_data["roc"] = {
                "fpr": roc_df["false_positive_rate"],
                "tpr": roc_df["true_positive_rate"],
                "auc": roc_df["auc"].iloc[0],
            }

        if "sklearn_predictions" in files:
            df = files["sklearn_predictions"]
            precision, recall, _ = precision_recall_curve(
                df["true_label"], df["confidence_positive"]
            )
            model_data["pr"] = {"precision": precision, "recall": recall}

        curves_data.append(model_data)

    return curves_data


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_curves_plot(curves_data):
    """Create interactive Plotly ROC and PR curves"""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve")
    )

    for data in curves_data:
        if "roc" in data:
            fig.add_trace(
                go.Scatter(
                    x=data["roc"]["fpr"],
                    y=data["roc"]["tpr"],
                    name=f"{data['model']} (AUC: {data['roc']['auc']:.3f})",
                    mode="lines",
                ),
                row=1,
                col=1,
            )

        if "pr" in data:
            fig.add_trace(
                go.Scatter(
                    x=data["pr"]["recall"],
                    y=data["pr"]["precision"],
                    name=data["model"],
                    mode="lines",
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_curves_plot(curves_data):
    """Create static Matplotlib ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(curves_data)))

    # Plot ROC curves
    for data, color in zip(curves_data, colors):
        if "roc" in data:
            ax1.plot(
                data["roc"]["fpr"],
                data["roc"]["tpr"],
                label=f"{data['model']} (AUC: {data['roc']['auc']:.3f})",
                color=color,
            )

    ax1.set_title("ROC Curve", pad=20, fontsize=14, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend()

    # Plot PR curves
    for data, color in zip(curves_data, colors):
        if "pr" in data:
            ax2.plot(
                data["pr"]["recall"],
                data["pr"]["precision"],
                label=data["model"],
                color=color,
            )

    ax2.set_title("Precision-Recall Curve", pad=20, fontsize=14, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend()

    plt.tight_layout()
    return fig


def curves_component(data):
    """Component for displaying ROC and PR curves"""

    st.markdown("##### Model Performance Curves")
    st.markdown(
        """
    Visualize model performance through ROC and Precision-Recall curves.
    These curves help understand the trade-offs in model predictions.
    """
    )

    # Prepare data
    curves_data = prepare_curves_data(data)

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_curves_plot(curves_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_curves_plot(curves_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About These Curves"):
        st.markdown(
            """
            **ROC Curve**:
            - Shows trade-off between true and false positive rates
            - AUC closer to 1.0 indicates better performance
            
            **Precision-Recall Curve**:
            - Shows trade-off between precision and recall
            - Useful for imbalanced datasets
            - Higher curve indicates better performance
        """
        )
