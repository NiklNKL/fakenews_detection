import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


def create_static_pie_chart(data, title, show_percents=True):
    """
    Helper function to create a static pie chart with matplotlib with labels inside
    """
    # Calculate total for percentage computation
    total = sum(data["count"])

    # Separate data for custom formatting
    values = data["count"]
    categories = data["label_names"]
    percentages = [(count / total) * 100 for count in values]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Create pie chart without labels first
    wedges, _ = ax.pie(
        values, colors=["#4ECB71", "#FF6B6B"], startangle=90, labels=None
    )

    # Add custom labels inside the wedges
    for i, (category, value, percentage) in enumerate(
        zip(categories, values, percentages)
    ):
        # Calculate angle for label placement
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2
        angle_rad = np.deg2rad(angle)

        # Calculate label position (use smaller distance to place inside)
        label_distance = 0.5  # This value determines how far from center (0 to 1)
        x = label_distance * np.cos(angle_rad)
        y = label_distance * np.sin(angle_rad)

        # Create multi-line label
        label_text = f"{category}\n{value:,}\n({percentage:.1f}%)"

        # Add label with white text and slight shadow for better visibility
        ax.text(
            x,
            y,
            label_text,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="white",
            path_effects=[
                path_effects.withStroke(linewidth=3, foreground="black", alpha=0.2)
            ],
        )

    # Set title with enhanced styling
    ax.set_title(
        title,
        pad=20,
        size=16,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=3.0),
    )

    # Equal aspect ratio
    ax.axis("equal")

    # Add some padding around the plot
    plt.tight_layout(pad=2.0)

    return fig


def create_pie_chart(data, title, show_percents=True):
    """
    Helper function to create a pie chart with percentages and counts
    """
    # Calculate total for percentage computation
    total = sum(data["count"])

    # Create labels with both count and percentage
    labels = [
        f"{label}<br>{count:,} ({(count/total)*100:.1f}%)"
        for label, count in zip(data["label_names"], data["count"])
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=data["count"],
                textposition="inside",
                textinfo="label",
                showlegend=False,
                marker=dict(
                    colors=["#4ECB71", "#FF6B6B"]
                ),  # Green for real, Red for fake
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)), height=400
    )

    return fig


def dataset_distribution_component(data):
    """
    Component for visualizing dataset distributions with pie charts
    """
    st.header("Dataset Distribution Analysis")

    # Create columns for all pie charts
    col1, col2, col3, col4 = st.columns(4)

    # Original dataset
    original_data = data[data["dataset"] == "original"]

    if st.session_state.use_static_plots:
        # Static plots with matplotlib
        fig_original = create_static_pie_chart(original_data, "Original Dataset")
        with col1:
            st.pyplot(fig_original)
            plt.close(fig_original)

        # Training splits
        splits = ["train", "valid", "test"]
        cols = [col2, col3, col4]

        for split, col in zip(splits, cols):
            split_data = data[data["dataset"] == split]
            fig = create_static_pie_chart(split_data, f"{split.title()} Set")
            with col:
                st.pyplot(fig)
                plt.close(fig)
    else:
        # Interactive plots with plotly
        fig_original = create_pie_chart(original_data, "Original Dataset")
        with col1:
            st.plotly_chart(fig_original, use_container_width=True)

        # Training splits
        splits = ["train", "valid", "test"]
        cols = [col2, col3, col4]

        for split, col in zip(splits, cols):
            split_data = data[data["dataset"] == split]
            fig = create_pie_chart(split_data, f"{split.title()} Set")
            with col:
                st.plotly_chart(fig, use_container_width=True)

    # Statistics and Interpretation in two columns below the charts
    col_stats, col_interp = st.columns(2)

    with col_stats:
        with st.expander("📊 Dataset Statistics"):
            st.subheader("Original Dataset")
            st.dataframe(
                original_data[["label_names", "count"]]
                .set_index("label_names")
                .round(4)
            )

            st.subheader("Split Statistics")
            split_stats = data[data["dataset"] != "original"]
            split_stats_pivot = split_stats.pivot(
                index="dataset", columns="label_names", values=["count"]
            ).round(4)
            st.dataframe(split_stats_pivot)

    with col_interp:
        with st.expander("ℹ️ Dataset Analysis"):
            total = original_data["count"].sum()
            real_pct = original_data[original_data["label_names"] == "real"][
                "proportion"
            ].iloc[0]
            fake_pct = original_data[original_data["label_names"] == "fake"][
                "proportion"
            ].iloc[0]

            st.subheader("Original Dataset Composition")
            st.markdown(
                f"""
            - Total samples: {total:,}
            - The dataset shows a relatively balanced distribution:
                - Real news: {real_pct:.1%} of the dataset
                - Fake news: {fake_pct:.1%} of the dataset
            - This balance is beneficial for model training as it reduces potential bias
            """
            )

            st.subheader("Split Analysis")
            train_total = data[data["dataset"] == "train"]["count"].sum()
            valid_total = data[data["dataset"] == "valid"]["count"].sum()
            test_total = data[data["dataset"] == "test"]["count"].sum()

            st.markdown(
                f"""
            **Sample Distribution:**
            - Training set: {train_total:,} samples ({train_total/total:.1%})
            - Validation set: {valid_total:,} samples ({valid_total/total:.1%})
            - Test set: {test_total:,} samples ({test_total/total:.1%})
            
            **Key Observations:**
            - All splits maintain similar class distribution ratios
            - The training set contains the majority of the data
            - Validation and test sets are appropriately sized for model evaluation
            
            **Class Balance:**
            - Each split maintains approximately 55/45 ratio between fake and real news
            - This consistent distribution helps ensure:
                - Reliable model training
                - Representative validation results
                - Accurate test set evaluation
            """
            )
