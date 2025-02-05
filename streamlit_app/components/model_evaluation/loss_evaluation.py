import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_loss_plot(data):
    """Create interactive Plotly loss plot"""
    fig = go.Figure()

    for model, files in data.items():
        if "training_logs" in files:
            df = files["training_logs"]
            fig.add_trace(
                go.Scatter(
                    x=df["step"], y=df["loss"], name=f"{model} (Train)", mode="lines"
                )
            )
        if "eval_logs" in files:
            df = files["eval_logs"]
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["loss"],
                    name=f"{model} (Eval)",
                    line=dict(dash="dash"),
                )
            )

    fig.update_layout(
        title="Training and Evaluation Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        height=500,
        template="plotly_white",
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_loss_plot(data):
    """Create static Matplotlib loss plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))

    for (model, files), color in zip(data.items(), colors):
        if "training_logs" in files:
            df = files["training_logs"]
            ax.plot(df["step"], df["loss"], label=f"{model} (Train)", color=color)
        if "eval_logs" in files:
            df = files["eval_logs"]
            ax.plot(
                df["step"],
                df["loss"],
                label=f"{model} (Eval)",
                color=color,
                linestyle="--",
            )

    # Customize plot
    ax.set_title("Training and Evaluation Loss", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)

    # Add grid and remove spines
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    plt.tight_layout()
    return fig


def loss_plots_component(data):
    """Component for displaying training and evaluation loss plots"""

    st.markdown(
        """
        This component shows the training and evaluation loss curves over time.
        It helps visualize model convergence and potential overfitting.
    """
    )

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_loss_plot(data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_loss_plot(data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Loss Plots"):
        st.markdown(
            """
            - **Training Loss**: Shows how well the model is fitting the training data
            - **Evaluation Loss**: Shows model performance on validation data
            - **Gap Analysis**: Large gap between curves may indicate overfitting
            - **Convergence**: Flattening curves suggest model convergence
        """
        )
