import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import numpy as np


root_path = Path(__file__).resolve().parent.parent.parent

# Load dependency labels from JSON file
with open(
    f"{root_path}/components/dataset_analysis/dependency_labels.json", "r"
) as file:
    DEPENDENCY_LABELS = json.load(file)


@st.cache_data(show_spinner="Preparing dependency data...")
def prepare_dependency_data(df):
    """
    Prepare dependency data for visualization
    """
    dependency_columns = [
        col for col in df.columns if col not in ["id", "label", "label_names"]
    ]

    df_fake = df[df["label_names"] == "fake"][dependency_columns].sum()
    df_real = df[df["label_names"] == "real"][dependency_columns].sum()

    df_combined = pd.DataFrame({"Fake News": df_fake, "Real News": df_real})

    return df_combined


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_dependency_plot(df_combined, language="en", top_n=10):
    """
    Generate interactive Plotly plot for dependency counts
    """
    if top_n:
        total_counts = df_combined["Fake News"] + df_combined["Real News"]
        df_combined = df_combined.loc[total_counts.nlargest(top_n).index]

    df_combined.index = [
        DEPENDENCY_LABELS.get(col, {}).get(language, {}).get("readable", col)
        for col in df_combined.index
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_combined.index,
            y=df_combined["Fake News"],
            name="Fake News",
            marker_color="#FF6B6B",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_combined.index,
            y=df_combined["Real News"],
            name="Real News",
            marker_color="#4ECB71",
        )
    )

    fig.update_layout(
        title=f"{'Dependency Counts' if language == 'en' else 'Abhängigkeitszähler'}: Fake vs Real News",
        xaxis_title=f"{'Dependency Type' if language == 'en' else 'Abhängigkeitstyp'}",
        yaxis_title=f"{'Count' if language == 'en' else 'Anzahl'}",
        barmode="group",
        height=600,
        template="plotly_white",
        xaxis_tickangle=-45,
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_dependency_plot(df_combined, language="en", top_n=10):
    """
    Generate static Matplotlib plot for dependency counts
    """
    if top_n:
        total_counts = df_combined["Fake News"] + df_combined["Real News"]
        df_combined = df_combined.loc[total_counts.nlargest(top_n).index]

    df_combined.index = [
        DEPENDENCY_LABELS.get(col, {}).get(language, {}).get("readable", col)
        for col in df_combined.index
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the width of each bar and positions of the bars
    width = 0.35
    x = np.arange(len(df_combined.index))

    # Create bars
    rects1 = ax.bar(
        x - width / 2,
        df_combined["Fake News"],
        width,
        label="Fake News",
        color="#FF6B6B",
        alpha=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        df_combined["Real News"],
        width,
        label="Real News",
        color="#4ECB71",
        alpha=0.8,
    )

    # Customize the plot
    title = "Dependency Counts" if language == "en" else "Abhängigkeitszähler"
    ax.set_title(f"{title}: Fake vs Real News", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel(
        f"{'Dependency Type' if language == 'en' else 'Abhängigkeitstyp'}",
        fontsize=12,
        labelpad=10,
    )
    ax.set_ylabel(
        f"{'Count' if language == 'en' else 'Anzahl'}", fontsize=12, labelpad=10
    )

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df_combined.index, rotation=45, ha="right")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{int(height):,}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    return fig


def dependency_analysis_component(df):
    """
    Streamlit component for dependency analysis
    """
    st.header("Dependency Analysis")
    df_combined = prepare_dependency_data(df)
    col1, col2 = st.columns(2)

    with col1:
        language = st.radio(
            "Select Language / Sprache", ["English", "Deutsch"], horizontal=True
        )
        language_code = "en" if language == "English" else "de"

    with col2:
        top_n_label = (
            "Show Top N Dependencies"
            if language_code == "en"
            else "Top N Abhängigkeiten anzeigen"
        )
        top_n = st.slider(
            top_n_label, min_value=5, max_value=len(df_combined), value=10
        )

    # Create and display plot based on selected type
    if st.session_state.use_static_plots:
        fig = create_static_dependency_plot(
            df_combined, language=language_code, top_n=top_n
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_dependency_plot(
            df_combined, language=language_code, top_n=top_n
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(
        f"{'More Information' if language_code == 'en' else 'Weitere Informationen'}"
    ):
        if language_code == "en":
            st.markdown(
                """
            ### Understanding Dependency Analysis

            Dependency analysis is a linguistic technique that examines the grammatical relationships between words in a sentence. In our context, we're comparing how these dependencies differ between fake and real news.

            #### Why is this Important?
            - Different types of dependencies can reveal linguistic patterns
            - May indicate structural differences in fake vs. real news writing
            - Provides insights into grammatical complexity and style
            """
            )
        else:
            st.markdown(
                """
            ### Verstehen der Abhängigkeitsanalyse

            Die Abhängigkeitsanalyse ist eine linguistische Technik, die die grammatikalischen Beziehungen zwischen Wörtern in einem Satz untersucht. In unserem Kontext vergleichen wir, wie sich diese Abhängigkeiten zwischen Fake- und Real-News unterscheiden.

            #### Warum ist dies wichtig?
            - Verschiedene Abhängigkeitstypen können linguistische Muster aufdecken
            - Kann strukturelle Unterschiede im Schreibstil von Fake- und Real-News zeigen
            - Bietet Einblicke in grammatikalische Komplexität und Stil
            """
            )

    with st.expander(
        f"{'Dependency Types Explained' if language_code == 'en' else 'Abhängigkeitstypen erklärt'}"
    ):
        cols = st.columns(3)
        df_combined["total"] = df_combined["Fake News"] + df_combined["Real News"]
        df_combined = df_combined.sort_values(by="total", ascending=False)
        top_dependencies = df_combined.index.tolist()[:top_n]
        for i, dep in enumerate(top_dependencies):
            with cols[i % 3]:
                try:
                    details = DEPENDENCY_LABELS[dep][language_code]
                except KeyError:
                    continue
                st.markdown(f"### {details['readable']}")
                st.markdown(f"**{details['description']}**")
                st.markdown(f"*{details['example']}*")
