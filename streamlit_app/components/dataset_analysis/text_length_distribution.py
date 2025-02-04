import streamlit as st
import plotly.express as px

def text_length_distribution_component(df):
    """
    Component for analyzing text length distribution with normalization option
    """
    st.header("Text Length Distribution Analysis")
    
    bins = [0, 20, 50, 100, 150, 200, 500, 1000, 2000, 3000, 4000, 5000]
    
    def classify_length(length):
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                return f"{bins[i]}-{bins[i + 1]}"
        if length >= bins[-1]:
            return f"{bins[-1]}+"
        return None
    
    df["length_bin"] = df["preprocessed_text"].apply(lambda x: classify_length(len(x) - x.count(" ")))

    bin_counts = df.groupby(['length_bin', 'label_names']).size().unstack(fill_value=0)
    
    normalize = st.checkbox("Normalize Distribution", value=True)
    
    if normalize:
        bin_counts = bin_counts.div(bin_counts.sum(axis=0), axis=1)
    
    bin_order = ["0-20", "20-50", "50-100", "100-150", "150-200", "200-500", 
                 "500-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000+"]
    bin_counts = bin_counts.reindex(bin_order)

    fig = px.bar(
        bin_counts.reset_index(), 
        x='length_bin', 
        y=['real', 'fake'], 
        title='Text Length Distribution by Category',
        labels={'value': 'Frequency', 'variable': 'Label'},
        color_discrete_map={'real': '#4ECB71', 'fake': '#FF6B6B'},
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title='Number of Characters in Text',
        yaxis_title='Relative Frequency' if normalize else 'Absolute Count',
        xaxis_tickangle=45
    )
    
    st.plotly_chart(fig)
    
    with st.expander("ℹ️ Interpretation"):
        st.markdown("""
        - This chart shows the distribution of text lengths across real and fake news
        - Normalization adjusts for the total number of documents in each category
        - Helps understand if text length differs between real and fake news
        """)