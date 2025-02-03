import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path

root_path = Path().resolve().parent

# Load dependency labels from JSON file
with open(f'{root_path}/streamlit_app/components/dataset_analysis/dependency_labels.json', 'r') as file:
    DEPENDENCY_LABELS = json.load(file)

@st.cache_data(show_spinner="Preparing dependency data...")
def prepare_dependency_data(df):
    """
    Prepare dependency data for visualization
    """
    dependency_columns = [col for col in df.columns if col not in ['id', 'label', "label_names"]]
    
    df_fake = df[df['label_names'] == "fake"][dependency_columns].sum()
    df_real = df[df['label_names'] == "real"][dependency_columns].sum()
    
    df_combined = pd.DataFrame({
        'Fake News': df_fake,
        'Real News': df_real
    })
    
    return df_combined

def generate_dependency_plot(df_combined, language='en', top_n=10):
    """
    Generate interactive plot for dependency counts
    """
    if top_n:
        total_counts = df_combined['Fake News'] + df_combined['Real News']
        df_combined = df_combined.loc[total_counts.nlargest(top_n).index]
    
    df_combined.index = [
        DEPENDENCY_LABELS.get(col, {}).get(language, {}).get('readable', col) 
        for col in df_combined.index
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_combined.index, 
        y=df_combined['Fake News'], 
        name='Fake News',
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        x=df_combined.index, 
        y=df_combined['Real News'], 
        name='Real News',
        marker_color='#4ECB71'
    ))
    
    fig.update_layout(
        title=f"{'Dependency Counts' if language == 'en' else 'Abhängigkeitszähler'}: Fake vs Real News",
        xaxis_title=f"{'Dependency Type' if language == 'en' else 'Abhängigkeitstyp'}",
        yaxis_title=f"{'Count' if language == 'en' else 'Anzahl'}",
        barmode='group',
        height=600,
        template='plotly_white',
        xaxis_tickangle=-45,
    )
    
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
            "Select Language / Sprache",
            ["English", "Deutsch"],
            horizontal=True
        )
        language_code = 'en' if language == 'English' else 'de'
    
    with col2:
        top_n_label = 'Show Top N Dependencies' if language_code == 'en' else 'Top N Abhängigkeiten anzeigen'
        top_n = st.slider(
            top_n_label, 
            min_value=5, 
            max_value=len(df_combined), 
            value=10
        )
    
    fig = generate_dependency_plot(
        df_combined, 
        language=language_code,
        top_n=top_n
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander(f"{'More Information' if language_code == 'en' else 'Weitere Informationen'}"):
        if language_code == 'en':
            st.markdown("""
            ### Understanding Dependency Analysis

            Dependency analysis is a linguistic technique that examines the grammatical relationships between words in a sentence. In our context, we're comparing how these dependencies differ between fake and real news.

            #### Why is this Important?
            - Different types of dependencies can reveal linguistic patterns
            - May indicate structural differences in fake vs. real news writing
            - Provides insights into grammatical complexity and style
            """)
        else:
            st.markdown("""
            ### Verstehen der Abhängigkeitsanalyse

            Die Abhängigkeitsanalyse ist eine linguistische Technik, die die grammatikalischen Beziehungen zwischen Wörtern in einem Satz untersucht. In unserem Kontext vergleichen wir, wie sich diese Abhängigkeiten zwischen Fake- und Real-News unterscheiden.

            #### Warum ist dies wichtig?
            - Verschiedene Abhängigkeitstypen können linguistische Muster aufdecken
            - Kann strukturelle Unterschiede im Schreibstil von Fake- und Real-News zeigen
            - Bietet Einblicke in grammatikalische Komplexität und Stil
            """)
    
    with st.expander(f"{'Dependency Types Explained' if language_code == 'en' else 'Abhängigkeitstypen erklärt'}"):
        cols = st.columns(3)
        top_dependencies = df_combined.index.tolist()[:9]
        for i, dep in enumerate(top_dependencies):
            with cols[i % 3]:
                try:
                    details = DEPENDENCY_LABELS[dep][language_code]
                except KeyError:
                    continue
                st.markdown(f"### {details['readable']}")
                st.markdown(f"**{details['description']}**")
                st.markdown(f"*{details['example']}*")
    
    with st.expander(f"{'Detailed Dependency Counts' if language_code == 'en' else 'Detaillierte Abhängigkeitszähler'}"):
        st.dataframe(df_combined)
