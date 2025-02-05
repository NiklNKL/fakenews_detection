import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np


def format_time(seconds):
    """Convert seconds to formatted string (hours, minutes, seconds)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def prepare_time_data(data):
    """Prepare time data for visualization"""
    times = {
        model: files["summary_logs"]["train_runtime"].sum()
        for model, files in data.items()
        if "summary_logs" in files
    }

    times_in_minutes = {model: time / 60 for model, time in times.items()}
    hover_texts = {model: format_time(time) for model, time in times.items()}

    return {
        "models": list(times_in_minutes.keys()),
        "times": list(times_in_minutes.values()),
        "hover_texts": list(hover_texts.values()),
    }


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_time_plot(plot_data):
    """Create interactive Plotly bar plot"""
    fig = px.bar(
        x=plot_data["models"],
        y=plot_data["times"],
        title="Training Time Comparison",
        labels={"x": "Model", "y": "Total Training Time (minutes)"},
        text=plot_data["hover_texts"],
    )

    fig.update_layout(height=500, xaxis_tickangle=-45, template="plotly_white")

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_time_plot(plot_data):
    """Create static Matplotlib bar plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(plot_data["models"], plot_data["times"])

    # Customize plot
    ax.set_title("Training Time Comparison", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Total Training Time (minutes)", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar, time_text in zip(bars, plot_data["hover_texts"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            time_text,
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Add grid and remove spines
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def training_time_component(data):
    """Component for displaying training time comparison"""

    st.markdown(
        """
        Compare training time across different models.
        This helps understand the computational efficiency of each approach.
    """
    )

    # Prepare data
    plot_data = prepare_time_data(data)

    # Create and display the appropriate plot
    if st.session_state.use_static_plots:
        fig = create_static_time_plot(plot_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_time_plot(plot_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Training Time"):
        st.markdown(
            """
            - Shows total training time for each model
            - Helps evaluate computational efficiency
            - Consider trade-off between time and performance
            - Useful for resource planning
            
            System used: 
            - **CPU:** 13th Gen Intel(R) Core(TM) i5-13600K   3.50 GHz
            - **GPU:** NVIDIA GeForce RTX 3080
            - **RAM:** 32GB DDR5 5600MT/s
        """
        )
