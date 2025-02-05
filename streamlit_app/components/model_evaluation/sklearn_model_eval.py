import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def format_time(seconds):
    """Convert seconds to formatted string (hours, minutes, seconds)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def prepare_bert_data(bert_data):
    """Prepare BERT model data"""
    metrics = ["accuracy", "f1", "precision", "recall"]
    models = {}
    for metric in metrics:
        values = {
            model: files["sklearn_metrics"][metric].iloc[0]
            for model, files in bert_data.items()
            if "sklearn_metrics" in files
        }
        sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
        models[metric] = sorted_values

    times = {
        model: files["summary_logs"]["train_runtime"].sum()
        for model, files in bert_data.items()
        if "summary_logs" in files
    }

    models_set = set(model for values in models.values() for model, _ in values)
    rows = []
    for model in models_set:
        row = {
            "Model": model,
            "Dataset": "Test",
            "Accuracy": np.nan,
            "F1": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
        }
        for metric, values in models.items():
            for m, score in values:
                if m == model:
                    row[metric.capitalize()] = score
        rows.append(row)
    for row in rows:
        row["Training_Time"] = times[row["Model"]]

    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Dataset",
            "Accuracy",
            "F1",
            "Precision",
            "Recall",
            "Training_Time",
        ],
    )


def create_metrics_tables(df):
    """Create performance metrics tables"""
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    df_filtered = df[df["Dataset"] == "Test"]
    models = {
        metric: df_filtered.sort_values(by=metric, ascending=False)["Model"].tolist()
        for metric in metrics
    }

    cols = st.columns(len(metrics))
    for metric, col in zip(metrics, cols):
        with col:
            st.markdown(f"### {metric} Score")
            table_data = []
            for rank, model in enumerate(models[metric], 1):
                train_score = df.loc[
                    (df["Model"] == model) & (df["Dataset"] == "Train"), metric
                ].values
                test_score = df.loc[
                    (df["Model"] == model) & (df["Dataset"] == "Test"), metric
                ].values[0]

                train_score = round(train_score[0], 4) if len(train_score) > 0 else None
                overfitting = (
                    "❓"
                    if train_score is None
                    else "⚠️" if train_score - test_score > 0.05 else "✅"
                )

                table_data.append((rank, model, train_score, test_score, overfitting))

            df_table = pd.DataFrame(
                table_data, columns=["Place", "Model", "Train", "Test", "Overfitting"]
            )
            st.table(df_table.drop(columns=["Place"]))


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_confusion_matrix(model, cm_data):
    """Create interactive Plotly confusion matrix"""
    cm_sum = sum(sum(row) for row in cm_data)
    cm_percentages = (
        [[(val / cm_sum) * 100 for val in row] for row in cm_data]
        if cm_sum > 0
        else [[0, 0], [0, 0]]
    )

    annotations = [
        [f"{int(v)}<br>({p:.1f}%)" for v, p in zip(row, prow)]
        for row, prow in zip(cm_data, cm_percentages)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_data,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title=model,
        height=400,
        width=400,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template="plotly_white",
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_confusion_matrix(model, cm_data):
    """Create static Matplotlib confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_sum = sum(sum(row) for row in cm_data)
    cm_percentages = [[(val / cm_sum) * 100 for val in row] for row in cm_data]

    sns.heatmap(cm_data, annot=True, cmap="Blues", cbar=False, ax=ax)

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({cm_percentages[i][j]:.1f}%)",
                ha="center",
                va="center",
            )

    ax.set_title(model, pad=20)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_time_plot(times_data):
    """Create interactive Plotly time plot"""
    fig = px.bar(
        x=list(times_data["models"]),
        y=list(times_data["times"]),
        title="Training Time Comparison",
        labels={"x": "Model", "y": "Total Training Time (minutes)"},
        text=list(times_data["hover_texts"]),
    )

    fig.update_layout(height=500, xaxis_tickangle=-45, template="plotly_white")

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_time_plot(times_data):
    """Create static Matplotlib time plot"""
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(times_data["models"], times_data["times"])

    # Add value labels
    for bar, time_text in zip(bars, times_data["hover_texts"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            time_text,
            ha="center",
            va="bottom",
            rotation=0,
        )

    ax.set_title("Training Time Comparison", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Total Training Time (minutes)", fontsize=12)

    plt.xticks(rotation=45, ha="right")

    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def sklearn_model_performance_component(df, bert_data):
    """Component to visualize model performance metrics."""

    st.sidebar.markdown("### Options")
    include_bert = st.sidebar.toggle("Include BERT Models", value=False)

    if include_bert:
        bert_df = prepare_bert_data(bert_data)
        df = pd.concat([df, bert_df], ignore_index=True)

    st.header("Sklearn Model Performance")
    st.write(
        "All of the following non-BERT models used the TF-IDF vectorizer with a maximum of 10000 features."
    )

    # Create metrics tables
    create_metrics_tables(df)

    with st.expander("ℹ️ About Performance Metrics"):
        st.markdown(
            """
            - **Accuracy**: Overall correctness of the model.
            - **Precision**: TP / (TP + FP), measures correctness of positive predictions.
            - **Recall**: TP / (TP + FN), measures completeness of positive predictions.
            - **F1-Score**: Harmonic mean of Precision and Recall.
        """
        )

    # Confusion Matrices
    st.markdown(
        """
        ## Confusion Matrices
        This component shows confusion matrices for each model, helping visualize 
        true positives, true negatives, false positives, and false negatives.
    """
    )

    # Create confusion matrices grid
    for i in range(0, len(df["Model"].unique()), 3):
        cols = st.columns(min(3, len(df["Model"].unique()) - i))
        for model, col in zip(df["Model"].unique()[i : i + 3], cols):
            with col:
                model_df = df[(df["Model"] == model) & (df["Dataset"] == "Test")]
                if not model_df.empty and all(
                    col in model_df.columns and pd.notna(model_df.iloc[0][col])
                    for col in ["tp", "tn", "fp", "fn"]
                ):
                    cm = [
                        [model_df.iloc[0]["tn"], model_df.iloc[0]["fp"]],
                        [model_df.iloc[0]["fn"], model_df.iloc[0]["tp"]],
                    ]

                    if st.session_state.use_static_plots:
                        fig = create_static_confusion_matrix(model, cm)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        fig = create_dynamic_confusion_matrix(model, cm)
                        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Confusion Matrix"):
        st.markdown(
            """
            The confusion matrix shows:
            
            - **True Negatives** (top-left): Correctly predicted negatives
            - **False Positives** (top-right): Incorrectly predicted positives
            - **False Negatives** (bottom-left): Incorrectly predicted negatives
            - **True Positives** (bottom-right): Correctly predicted positives
            
            Each cell shows:
            - The absolute count
            - The percentage of total predictions (in parentheses)
            
            💡 A good model should have high numbers in the diagonal (top-left to bottom-right)
            and low numbers in the off-diagonal.
        """
        )

    # Training Time Analysis
    st.markdown(
        """
        ## Training Time Analysis
        Compare the time taken by each model for training.
    """
    )

    times = {
        model: df.loc[df["Model"] == model, "Training_Time"].sum()
        for model in df["Model"].unique()
    }
    times_in_minutes = {
        model: time / 60 for model, time in times.items() if pd.notnull(time)
    }
    hover_texts = {
        model: format_time(time) for model, time in times.items() if pd.notnull(time)
    }

    times_data = {
        "models": list(times_in_minutes.keys()),
        "times": list(times_in_minutes.values()),
        "hover_texts": list(hover_texts.values()),
    }

    if st.session_state.use_static_plots:
        fig = create_static_time_plot(times_data)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_time_plot(times_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Training Time"):
        st.markdown(
            """
            - Shows total training time for each model
            - Helps evaluate computational efficiency
            - Consider trade-off between time and performance
            - Useful for resource planning
        """
        )
