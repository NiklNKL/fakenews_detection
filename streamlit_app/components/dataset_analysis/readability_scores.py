import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def get_metric_info(metric):
    """
    Returns interpretation and formula information for each metric
    """
    info = {
        "flesch_reading_ease": {
            "formula": "206.835 - 1.015(total words/total sentences) - 84.6(total syllables/total words)",
            "interpretation": """
            ### Flesch Reading Ease Score
            
            **Score Interpretation:**
            - 90-100: Very Easy (5th grade)
            - 80-89: Easy (6th grade)
            - 70-79: Fairly Easy (7th grade)
            - 60-69: Standard (8th-9th grade)
            - 50-59: Fairly Difficult (10th-12th grade)
            - 30-49: Difficult (College)
            - 0-29: Very Difficult (College Graduate)
            
            **Key Points:**
            - Higher scores indicate easier readability
            - Most business documents aim for 60-70
            - Popular novels typically score 70-80
            """,
            "description": """
            The Flesch Reading Ease Score is one of the oldest and most accurate readability formulas. 
            It considers both sentence length and word complexity (syllable count) to determine how difficult a text is to understand.
            """,
        },
        "smog_index": {
            "formula": "1.0430 × √(number of polysyllables × (30/number of sentences)) + 3.1291",
            "interpretation": """
            ### SMOG Index
            
            **Score Interpretation:**
            - 6-7: 6th-7th grade
            - 8-9: 8th-9th grade
            - 10-11: 10th-11th grade
            - 12-13: 12th grade-college
            - 14+: College graduate
            
            **Key Points:**
            - Focuses on polysyllabic words
            - Considered highly accurate for health materials
            - Generally predicts 100% comprehension
            """,
            "description": """
            The SMOG (Simple Measure of Gobbledygook) Index estimates the years of education needed to understand a text. 
            It's particularly popular in healthcare and medical communication due to its accuracy in predicting comprehension.
            """,
        },
        "dale_chall_score": {
            "formula": "0.1579 × (difficult words/words × 100) + 0.0496 × (words/sentences)",
            "interpretation": """
            ### Dale-Chall Score
            
            **Score Interpretation:**
            - 4.9 or lower: 4th grade or lower
            - 5.0–5.9: 5th-6th grade
            - 6.0–6.9: 7th-8th grade
            - 7.0–7.9: 9th-10th grade
            - 8.0–8.9: 11th-12th grade
            - 9.0–9.9: College
            - 10.0+: College graduate
            
            **Key Points:**
            - Uses a list of 3,000 familiar words
            - Counts "difficult" words not in this list
            - Highly respected in education
            """,
            "description": """
            The Dale-Chall Score is unique because it uses a predefined list of familiar words rather than syllable count. 
            Words not on this list are considered 'difficult,' making it particularly useful for educational content assessment.
            """,
        },
    }
    return info[metric]


def prepare_distribution_data(data, metric):
    """
    Prepare data for distribution plots
    """
    fake_data = data[data["label_names"] == "fake"][metric]
    real_data = data[data["label_names"] == "real"][metric]

    x_range = np.linspace(
        min(fake_data.min(), real_data.min()),
        max(fake_data.max(), real_data.max()),
        200,
    )

    fake_kde = gaussian_kde(fake_data)
    real_kde = gaussian_kde(real_data)

    return {
        "x_range": x_range,
        "fake_kde": fake_kde(x_range),
        "real_kde": real_kde(x_range),
        "fake_data": fake_data,
        "real_data": real_data,
    }


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_distribution_plot(plot_data, metric, title):
    """
    Creates an interactive Plotly distribution plot
    """
    fig = go.Figure()

    # Add traces for fake and real news
    fig.add_trace(
        go.Scatter(
            x=plot_data["x_range"],
            y=plot_data["fake_kde"],
            name="Fake News",
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.2)",
            line=dict(color="#FF6B6B", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data["x_range"],
            y=plot_data["real_kde"],
            name="Real News",
            fill="tozeroy",
            fillcolor="rgba(78,203,113,0.2)",
            line=dict(color="#4ECB71", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="Density",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_distribution_plot(plot_data, metric, title):
    """
    Creates a static Matplotlib distribution plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distributions
    ax.fill_between(
        plot_data["x_range"],
        plot_data["fake_kde"],
        alpha=0.2,
        color="#FF6B6B",
        label="Fake News",
    )
    ax.plot(plot_data["x_range"], plot_data["fake_kde"], color="#FF6B6B", linewidth=2)

    ax.fill_between(
        plot_data["x_range"],
        plot_data["real_kde"],
        alpha=0.2,
        color="#4ECB71",
        label="Real News",
    )
    ax.plot(plot_data["x_range"], plot_data["real_kde"], color="#4ECB71", linewidth=2)

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Add grid and remove top/right spines
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def readability_scores_component(df):
    """
    Component for analyzing readability metrics
    """
    st.header("Readability Metrics Analysis")

    metrics = {
        "flesch_reading_ease": "Flesch Reading Ease",
        "smog_index": "SMOG Index",
        "dale_chall_score": "Dale-Chall Score",
    }

    # Create tabs
    tabs = st.tabs(list(metrics.values()))

    for tab, (metric, title) in zip(tabs, metrics.items()):
        with tab:
            # Prepare plot data
            plot_data = prepare_distribution_data(df, metric)

            # Create and display plot based on selected type
            if st.session_state.use_static_plots:
                fig = create_static_distribution_plot(
                    plot_data, metric, f"{title} Distribution"
                )
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig = create_dynamic_distribution_plot(
                    plot_data, metric, f"{title} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Display statistics and interpretation (rest remains the same)
            metric_info = get_metric_info(metric)

            col_stats, col_interp = st.columns(2)
            with col_stats:
                with st.expander("📊 Statistical Summary"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Fake News")
                        fake_stats = (
                            df[df["label_names"] == "fake"][metric].describe().round(2)
                        )
                        st.dataframe(pd.DataFrame(fake_stats))

                    with col2:
                        st.subheader("Real News")
                        real_stats = (
                            df[df["label_names"] == "real"][metric].describe().round(2)
                        )
                        st.dataframe(pd.DataFrame(real_stats))

            with col_interp:
                with st.expander("ℹ️ Understanding the Metric"):
                    st.markdown(metric_info["description"])

                    st.subheader("Formula")
                    st.code(metric_info["formula"])

                    st.markdown(metric_info["interpretation"])

                    fake_mean = df[df["label_names"] == "fake"][metric].mean()
                    real_mean = df[df["label_names"] == "real"][metric].mean()
                    diff = abs(fake_mean - real_mean)

                    st.subheader("Fake vs. Real News Analysis")
                    st.markdown(
                        f"""
                    **Average Scores:**
                    - Fake News: {fake_mean:.2f}
                    - Real News: {real_mean:.2f}
                    - Difference: {diff:.2f}
                    """
                    )
