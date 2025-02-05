import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def prepare_confusion_matrix_data(data):
    """Prepare confusion matrix data for visualization"""
    model_aliases = {
        "roberta-base": "RoBERTa",
        "bert-base-uncased": "BERT",
        "distilbert-base-uncased": "DistilBERT",
    }

    matrices_data = []
    for model, files in data.items():
        if "sklearn_confusion" in files:
            df = files["sklearn_confusion"]
            cm = df.iloc[:, 1:].values.reshape(2, 2)
            cm_sum = cm.sum()
            cm_percentages = cm / cm_sum * 100

            base_model_name = model.split(" ")[0]
            model_name = model_aliases.get(base_model_name, base_model_name)
            if "(with_peft)" in model:
                model_name += " with PEFT"
            else:
                model_name += " without PEFT"

            matrices_data.append(
                {
                    "model": model,
                    "display_name": model_name,
                    "matrix": cm,
                    "percentages": cm_percentages,
                }
            )

    return matrices_data


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_confusion_matrix(matrix_data):
    """Create interactive Plotly confusion matrix"""
    # Calculate annotations
    annotations = [
        [f"{int(v)}<br>({p:.1f}%)" for v, p in zip(row, prow)]
        for row, prow in zip(matrix_data["matrix"], matrix_data["percentages"])
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_data["matrix"],
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
        title=matrix_data["display_name"],
        height=400,
        width=400,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template="plotly_white",
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_confusion_matrix(matrix_data):
    """Create static Matplotlib confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create heatmap
    sns.heatmap(
        matrix_data["matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Neg", "Pred Pos"],
        yticklabels=["Act Neg", "Act Pos"],
    )

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            percentage = matrix_data["percentages"][i, j]
            ax.text(j + 0.5, i + 0.7, f"({percentage:.1f}%)", ha="center", va="center")

    # Customize plot
    plt.title(matrix_data["display_name"], pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    return fig


def confusion_matrix_component(data):
    """Component for displaying confusion matrices for each model"""

    st.markdown(
        """
        This component shows confusion matrices for each model, helping visualize 
        true positives, true negatives, false positives, and false negatives.
    """
    )

    # Prepare data
    matrices_data = prepare_confusion_matrix_data(data)

    # Display matrices in a grid (max 3 per row)
    for i in range(0, len(matrices_data), 3):
        cols = st.columns(min(3, len(matrices_data) - i))
        for matrix_data, col in zip(matrices_data[i : i + 3], cols):
            with col:
                if st.session_state.use_static_plots:
                    fig = create_static_confusion_matrix(matrix_data)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    fig = create_dynamic_confusion_matrix(matrix_data)
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
