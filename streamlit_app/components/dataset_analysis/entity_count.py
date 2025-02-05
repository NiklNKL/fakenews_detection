import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def prepare_entity_data(df: pd.DataFrame, entity_dict: dict, normalize: bool = False):
    """
    Prepare entity data for visualization
    """
    entity_columns = list(entity_dict.keys())
    df_entity_by_label = df.groupby("label_names")[entity_columns].sum()

    if normalize:
        df_entity_by_label = df_entity_by_label.div(
            df_entity_by_label.sum(axis=1), axis=0
        )

    return df_entity_by_label


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_entity_plot(display_data, entity_dict, selected_columns, normalize):
    """
    Create interactive Plotly bar plot for entity analysis
    """
    y_axis_title = "Percentage of Entities" if normalize else "Count"
    title_suffix = " (Normalized)" if normalize else ""

    fig = go.Figure(
        data=[
            go.Bar(
                name="Fake News",
                x=[entity_dict[col] for col in selected_columns],
                y=display_data.loc["fake", selected_columns],
                marker_color="#FF6B6B",
            ),
            go.Bar(
                name="Real News",
                x=[entity_dict[col] for col in selected_columns],
                y=display_data.loc["real", selected_columns],
                marker_color="#4ECB71",
            ),
        ]
    )

    fig.update_layout(
        title=f"Entity Distribution in Fake and Real News{title_suffix}",
        xaxis_title="Entity Type",
        yaxis_title=y_axis_title,
        barmode="group",
        height=600,
        xaxis_tickangle=-45,
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_entity_plot(display_data, entity_dict, selected_columns, normalize):
    """
    Create static Matplotlib bar plot for entity analysis
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(selected_columns))
    width = 0.35

    # Create bars
    rects1 = ax.bar(
        x - width / 2,
        display_data.loc["fake", selected_columns],
        width,
        label="Fake News",
        color="#FF6B6B",
        alpha=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        display_data.loc["real", selected_columns],
        width,
        label="Real News",
        color="#4ECB71",
        alpha=0.8,
    )

    # Customize the plot
    title_suffix = " (Normalized)" if normalize else ""
    y_axis_title = "Percentage of Entities" if normalize else "Count"

    ax.set_title(
        f"Entity Distribution in Fake and Real News{title_suffix}",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Entity Type", fontsize=12, labelpad=10)
    ax.set_ylabel(y_axis_title, fontsize=12, labelpad=10)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [entity_dict[col] for col in selected_columns], rotation=45, ha="right"
    )

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if normalize:
                label = f"{height:.1%}"
            else:
                label = f"{int(height):,}"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    autolabel(rects1)
    autolabel(rects2)

    # Adjust layout
    plt.tight_layout()

    return fig


def entity_analysis_component(df: pd.DataFrame):
    """
    Create an interactive Streamlit component for visualizing entity counts
    across fake and real news labels, with normalization option.
    """
    # Entity type mapping
    entity_dict = {
        "PERSON": "Person",
        "MONEY": "Money",
        "TIME": "Time",
        "GPE": "Geopolitical Entity (Country/City)",
        "CARDINAL": "Cardinal Numbers",
        "PRODUCT": "Product",
        "ORG": "Organization",
        "ORDINAL": "Ordinal Numbers",
        "FAC": "Facility",
        "EVENT": "Event",
        "NORP": "Nationalities/Religions/Political Groups",
        "WORK_OF_ART": "Works of Art",
        "LAW": "Laws",
        "QUANTITY": "Quantity",
        "LOC": "Location",
        "PERCENT": "Percentage",
        "LANGUAGE": "Language",
    }

    st.markdown("## Entity Counts in Fake and Real News")

    if "normalize_entities" not in st.session_state:
        st.session_state.normalize_entities = False

    normalize = st.toggle(
        "Normalize Data",
        value=st.session_state.normalize_entities,
        help="Normalize entity counts as a percentage of total entities for each news type",
    )
    st.session_state.normalize_entities = normalize

    # Prepare data
    available_entities = {entity_dict[col]: col for col in entity_dict.keys()}
    selected_columns = [available_entities[entity] for entity in entity_dict.values()]
    display_data = prepare_entity_data(df, entity_dict, normalize)

    # Create and display plot based on selected type
    if st.session_state.use_static_plots:
        fig = create_static_entity_plot(
            display_data, entity_dict, selected_columns, normalize
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = create_dynamic_entity_plot(
            display_data, entity_dict, selected_columns, normalize
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About Entity Types and Normalization"):
        st.markdown(
            """
        ### Entity Type Insights
        - **Entities** represent different types of named or numeric elements in text
        - Comparing entity distributions can reveal differences between fake and real news
        - Some entity types might be more prevalent in one news type vs. another
        
        ### Normalization
        - **Raw Count**: Shows absolute number of entities
        - **Normalized**: Shows percentage of total entities for each news type
        - Helps compare entity distribution regardless of total document count
        
        ### Common Entity Types
        - **Geopolitical Entities (GPE)**: Countries, cities, states
        - **Organizations (ORG)**: Companies, agencies, institutions
        - **Persons (PERSON)**: Individual people
        - **Events (EVENT)**: Named happenings
        """
        )
