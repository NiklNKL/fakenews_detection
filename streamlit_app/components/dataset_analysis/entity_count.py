import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def entity_analysis_component(df:pd.DataFrame):
    """
    Create an interactive Streamlit component for visualizing entity counts 
    across fake and real news labels, with normalization option.
    
    Args:
        data_path (str): Path to the parquet file containing entity count data
    """
    # Entity type mapping
    entity_dict = {
        'PERSON': 'Person', 
        'MONEY': 'Money', 
        'TIME': 'Time', 
        'GPE': 'Geopolitical Entity (Country/City)', 
        'CARDINAL': 'Cardinal Numbers', 
        'PRODUCT': 'Product', 
        'ORG': 'Organization', 
        'ORDINAL': 'Ordinal Numbers', 
        'FAC': 'Facility', 
        'EVENT': 'Event', 
        'NORP': 'Nationalities/Religions/Political Groups', 
        'WORK_OF_ART': 'Works of Art', 
        'LAW': 'Laws', 
        'QUANTITY': 'Quantity', 
        'LOC': 'Location', 
        'PERCENT': 'Percentage', 
        'LANGUAGE': 'Language'
    }

    entity_columns = list(entity_dict.keys())
    
    df_entity_by_label = df.groupby('label_names')[entity_columns].sum()

    if 'normalize_entities' not in st.session_state:
        st.session_state.normalize_entities = False

    def normalize_data(data):
        """Normalize data by dividing each value by the total count for that label"""
        return data.div(data.sum(axis=1), axis=0)

    st.markdown("## Entity Counts in Fake and Real News")
    
    normalize = st.toggle(
        "Normalize Data", 
        value=st.session_state.normalize_entities,
        help="Normalize entity counts as a percentage of total entities for each news type"
    )
    st.session_state.normalize_entities = normalize
    
    available_entities = {entity_dict[col]: col for col in entity_columns}
    selected_columns = [available_entities[entity] for entity in available_entities]
    
    display_data = df_entity_by_label.copy()
    y_axis_title = "Count"
    title_suffix = ""
    
    if normalize:
        display_data = normalize_data(display_data)
        y_axis_title = "Percentage of Entities"
        title_suffix = " (Normalized)"

    fig = go.Figure(data=[
        go.Bar(
            name='Fake News', 
            x=[entity_dict[col] for col in selected_columns], 
            y=display_data.loc[0, selected_columns],
            marker_color='red'
        ),
        go.Bar(
            name='Real News', 
            x=[entity_dict[col] for col in selected_columns], 
            y=display_data.loc[1, selected_columns],
            marker_color='green'
        )
    ])
    
    fig.update_layout(
        title=f'Entity Distribution in Fake and Real News{title_suffix}',
        xaxis_title='Entity Type',
        yaxis_title=y_axis_title,
        barmode='group',
        height=600,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ About Entity Types and Normalization"):
        st.markdown("""
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
        """)