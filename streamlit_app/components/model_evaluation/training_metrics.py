import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def prepare_metrics_data(data, selected_metrics):
    """Prepare performance metrics data"""
    metrics_data = []
    for model, files in data.items():
        if "eval_logs" in files:
            df = files["eval_logs"]
            for metric in selected_metrics:
                if metric in df.columns:
                    metrics_data.append(
                        {
                            "model": model,
                            "metric": metric,
                            "steps": df["step"],
                            "values": df[metric],
                        }
                    )
    return metrics_data


def prepare_gradient_data(data):
    """Prepare gradient norm data"""
    grad_data = []
    for model, files in data.items():
        if "training_logs" in files:
            df = files["training_logs"]
            if "grad_norm" in df.columns:
                grad_data.append(
                    {"model": model, "steps": df["step"], "values": df["grad_norm"]}
                )
    return grad_data


def prepare_speed_data(data):
    """Prepare training speed data"""
    speed_data = []
    speed_stats = {}

    for model, files in data.items():
        if "training_logs" in files:
            df = files["eval_logs"]
            if "samples_per_second" in df.columns:
                speeds = df["samples_per_second"]
                speed_stats[model] = {
                    "mean": speeds.mean(),
                    "min": speeds.min(),
                    "max": speeds.max(),
                }
                speed_data.append({"model": model, "speeds": speeds})

    return speed_data, speed_stats


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_metrics_plot(metrics_data):
    """Create interactive Plotly metrics plot"""
    fig = go.Figure()

    for data in metrics_data:
        fig.add_trace(
            go.Scatter(
                x=data["steps"],
                y=data["values"],
                name=f"{data['model']} - {data['metric']}",
                line=dict(dash="solid" if data["metric"] == "accuracy" else "dash"),
            )
        )

    fig.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Step",
        yaxis_title="Score",
        height=500,
        template="plotly_white",
    )
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_metrics_plot(metrics_data):
    """Create static Matplotlib metrics plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_data)))

    for data, color in zip(metrics_data, colors):
        linestyle = "-" if data["metric"] == "accuracy" else "--"
        ax.plot(
            data["steps"],
            data["values"],
            label=f"{data['model']} - {data['metric']}",
            color=color,
            linestyle=linestyle,
        )

    ax.set_title(
        "Performance Metrics Over Time", pad=20, fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_gradient_plot(grad_data):
    """Create interactive Plotly gradient plot"""
    fig = go.Figure()

    for data in grad_data:
        fig.add_trace(go.Scatter(x=data["steps"], y=data["values"], name=data["model"]))

    fig.update_layout(
        title="Gradient Norm Over Time",
        xaxis_title="Step",
        yaxis_title="Gradient Norm",
        height=500,
        template="plotly_white",
    )
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_gradient_plot(grad_data):
    """Create static Matplotlib gradient plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(grad_data)))

    for data, color in zip(grad_data, colors):
        ax.plot(data["steps"], data["values"], label=data["model"], color=color)

    ax.set_title("Gradient Norm Over Time", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Gradient Norm", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_speed_plot(speed_data):
    """Create interactive Plotly speed plot"""
    fig = go.Figure()

    for data in speed_data:
        fig.add_trace(
            go.Box(y=data["speeds"], name=data["model"], boxpoints="outliers")
        )

    fig.update_layout(
        title="Training Speed Distribution",
        yaxis_title="Samples per Second",
        height=500,
        template="plotly_white",
    )
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_speed_plot(speed_data):
    """Create static Matplotlib speed plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = range(len(speed_data))
    ax.boxplot(
        [data["speeds"] for data in speed_data],
        positions=positions,
        labels=[data["model"] for data in speed_data],
    )

    ax.set_title("Training Speed Distribution", pad=20, fontsize=14, fontweight="bold")
    ax.set_ylabel("Samples per Second", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def training_metrics_component(data):
    """Component for tracking various training metrics over time with selectable metrics."""

    st.markdown(
        """
        Track various training metrics over time to understand model behavior
        and performance during training.
    """
    )

    # Create tabs for different metric groups
    tab1, tab2, tab3 = st.tabs(
        ["Performance Metrics", "Training Dynamics", "Speed Metrics"]
    )

    # Performance Metrics Tab
    with tab1:
        col1, col2 = st.columns([7, 1])

        with col2:
            available_metrics = {
                "Accuracy": "accuracy",
                "F1 Score": "f1",
                "Precision": "precision",
                "Recall": "recall",
            }
            initial_selected_metrics = st.multiselect(
                "Select Metrics to Display",
                available_metrics.keys(),
                default="Accuracy",
            )
            selected_metrics = [
                available_metrics[metric] for metric in initial_selected_metrics
            ]

        with col1:
            if not selected_metrics:
                st.warning("Please select at least one metric to display.")
                return

            metrics_data = prepare_metrics_data(data, selected_metrics)

            if st.session_state.use_static_plots:
                fig = create_static_metrics_plot(metrics_data)
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig = create_dynamic_metrics_plot(metrics_data)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ About Performance Metrics"):
            st.markdown(
                """
                - **Accuracy**: Measures how often predictions match labels  
                - **F1 Score**: Balances precision and recall  
                - **Precision**: Measures the percentage of relevant predictions  
                - **Recall**: Measures how many relevant cases were retrieved  
                - **Higher scores indicate better model performance**
            """
            )

    # Training Dynamics Tab
    with tab2:
        grad_data = prepare_gradient_data(data)

        if st.session_state.use_static_plots:
            fig = create_static_gradient_plot(grad_data)
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = create_dynamic_gradient_plot(grad_data)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ About Training Dynamics"):
            st.markdown(
                """
                - **Gradient Norm**: Measures updates to model weights  
                - **Large gradients** may indicate instability  
                - **Flat gradients** suggest slow learning  
                - **Helps diagnose vanishing/exploding gradients issues**
            """
            )

    # Speed Metrics Tab
    with tab3:
        speed_data, speed_stats = prepare_speed_data(data)

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.use_static_plots:
                fig = create_static_speed_plot(speed_data)
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig = create_dynamic_speed_plot(speed_data)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Speed Statistics")
            for model, stats in speed_stats.items():
                st.markdown(
                    f"""
                    **{model}**:
                    - Mean: {stats['mean']:.2f} samples/s
                    - Min: {stats['min']:.2f} samples/s
                    - Max: {stats['max']:.2f} samples/s
                """
                )

        with st.expander("ℹ️ About Speed Metrics"):
            st.markdown(
                """
                - **Samples per second**: Measures model training speed  
                - **High variability** may indicate inefficient data loading  
                - **Optimizing speed** can improve training efficiency  
            """
            )
