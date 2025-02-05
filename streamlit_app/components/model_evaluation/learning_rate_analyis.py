import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def prepare_lr_data(data):
    """Prepare learning rate and loss data"""
    lr_data = []
    for model, files in data.items():
        if "training_logs" in files:
            df = files["training_logs"]
            lr_data.append(
                {
                    "model": model,
                    "steps": df["step"],
                    "learning_rate": df["learning_rate"],
                    "loss": df["loss"],
                }
            )
    return lr_data


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_lr_plot(lr_data):
    """Create interactive Plotly learning rate plot"""
    fig = go.Figure()

    for data in lr_data:
        # Add learning rate trace
        fig.add_trace(
            go.Scatter(
                x=data["steps"],
                y=data["learning_rate"],
                name=f"{data['model']} (Learning Rate)",
                line=dict(dash="solid"),
            )
        )

        # Add loss trace on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=data["steps"],
                y=data["loss"],
                name=f"{data['model']} (Loss)",
                line=dict(dash="dot"),
                yaxis="y2",
            )
        )

    # Update layout with secondary y-axis
    fig.update_layout(
        title="Learning Rate and Loss over Training Steps",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        yaxis2=dict(title="Loss", overlaying="y", side="right"),
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_lr_plot(lr_data):
    """Create static Matplotlib learning rate plot"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create second y-axis
    ax2 = ax1.twinx()

    # Calculate number of models for color distribution
    n_models = len(lr_data)
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))

    for data, color in zip(lr_data, colors):
        # Plot learning rate on first y-axis
        line1 = ax1.plot(
            data["steps"],
            data["learning_rate"],
            label=f"{data['model']} (LR)",
            color=color,
            linestyle="-",
        )

        # Plot loss on second y-axis
        line2 = ax2.plot(
            data["steps"],
            data["loss"],
            label=f"{data['model']} (Loss)",
            color=color,
            linestyle=":",
        )

    # Customize plot
    ax1.set_title(
        "Learning Rate and Loss over Training Steps",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)

    # Add grid and remove spines
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(0, 1.15),
        ncol=2,
    )

    plt.tight_layout()
    return fig


def learning_rate_analysis_component(data):
    """Component for analyzing learning rate behavior"""

    st.markdown(
        """
        Analyze learning rate behavior and its effects during training.
        This helps understand the optimization process and learning dynamics.
    """
    )

    # Prepare data
    lr_data = prepare_lr_data(data)

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_lr_plot(lr_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_lr_plot(lr_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Learning Rate Analysis"):
        st.markdown(
            """
            - **Learning Rate**: Shows the learning rate schedule during training
            - **Loss**: Shows how the model's loss changes during training
            - Important for understanding:
                - Learning rate scheduling effectiveness
                - Correlation between learning rate and loss
                - Training stability and convergence
                - Optimal learning rate ranges
        """
        )
