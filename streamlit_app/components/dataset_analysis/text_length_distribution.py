import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np


def prepare_length_distribution_data(df):
    """
    Prepare data for length distribution visualization
    """
    bins = [0, 20, 50, 100, 150, 200, 500, 1000, 2000, 3000, 4000, 5000]

    def classify_length(length):
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                return f"{bins[i]}-{bins[i + 1]}"
        if length >= bins[-1]:
            return f"{bins[-1]}+"
        return None

    df["length_bin"] = df["raw_text"].apply(
        lambda x: classify_length(len(x) - x.count(" "))
    )
    bin_counts = df.groupby(["length_bin", "label_names"]).size().unstack(fill_value=0)

    bin_order = [
        "0-20",
        "20-50",
        "50-100",
        "100-150",
        "150-200",
        "200-500",
        "500-1000",
        "1000-2000",
        "2000-3000",
        "3000-4000",
        "4000-5000",
        "5000+",
    ]

    return bin_counts.reindex(bin_order)


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_plot(bin_counts, normalize):
    """
    Create interactive Plotly plot
    """
    fig = px.bar(
        bin_counts.reset_index(),
        x="length_bin",
        y=["real", "fake"],
        title="Text Length Distribution by Category",
        labels={"value": "Frequency", "variable": "Label"},
        color_discrete_map={"real": "#4ECB71", "fake": "#FF6B6B"},
        barmode="group",
    )

    fig.update_layout(
        xaxis_title="Number of Characters in Text",
        yaxis_title="Relative Frequency" if normalize else "Absolute Count",
        xaxis_tickangle=45,
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_plot(bin_counts, normalize):
    """
    Create static Matplotlib plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.35
    x = np.arange(len(bin_counts.index))

    rects1 = ax.bar(
        x - width / 2,
        bin_counts["real"],
        width,
        label="Real",
        color="#4ECB71",
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )
    rects2 = ax.bar(
        x + width / 2,
        bin_counts["fake"],
        width,
        label="Fake",
        color="#FF6B6B",
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}" if normalize else f"{int(height):,}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    autolabel(rects1)
    autolabel(rects2)

    # Customize the plot
    ax.set_title(
        "Text Length Distribution by Category", pad=20, size=14, fontweight="bold"
    )
    ax.set_xlabel("Number of Characters in Text", size=12, labelpad=10)
    ax.set_ylabel(
        "Relative Frequency" if normalize else "Absolute Count", size=12, labelpad=10
    )

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(bin_counts.index, rotation=45, ha="right")

    # Customize grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Enhanced legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    return fig


def text_length_distribution_component(df):
    """
    Component for analyzing text length distribution with normalization option
    """
    st.header("Text Length Distribution Analysis")

    normalize = st.checkbox("Normalize Distribution", value=True)

    # Prepare data
    bin_counts = prepare_length_distribution_data(df)
    if normalize:
        bin_counts = bin_counts.div(bin_counts.sum(axis=0), axis=1)

    # Create and display plot based on selected type
    if st.session_state.use_static_plots:
        fig = create_static_plot(bin_counts, normalize)
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_plot(bin_counts, normalize)
        st.plotly_chart(fig)

    with st.expander("ℹ️ Interpretation"):
        st.markdown(
            """
        - This chart shows the distribution of text lengths across real and fake news
        - Normalization adjusts for the total number of documents in each category
        - Helps understand if text length differs between real and fake news
        """
        )
