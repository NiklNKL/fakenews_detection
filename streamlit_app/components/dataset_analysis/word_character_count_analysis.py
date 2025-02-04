import streamlit as st
import plotly.express as px

def word_character_count_analysis_component(df):
    """
    Component for analyzing word and character count statistics
    """
    st.header("Word and Character Count Analysis")
    
    df["word_count"] = df["raw_text"].apply(lambda x: len(x.split()))
    df["body_len"] = df["raw_text"].apply(lambda x: len(x) - x.count(" "))

    word_count_stats = df.groupby("label_names")["word_count"].describe()
    char_count_stats = df.groupby("label_names")["body_len"].describe()

    fig_word = px.box(
        df, 
        x="label_names", 
        y="word_count", 
        color="label_names",
        title="Word Count Distribution",
        color_discrete_map={'real': 'green', 'fake': 'red'},
        log_y=True
    )
    fig_word.update_layout(xaxis_title="Label", yaxis_title="Word Count (Log Scale)")
    
    fig_char = px.box(
        df, 
        x="label_names", 
        y="body_len", 
        color="label_names",
        title="Character Count Distribution",
        color_discrete_map={'real': 'green', 'fake': 'red'},
        log_y=True
    )
    fig_char.update_layout(xaxis_title="Label", yaxis_title="Character Count (Log Scale)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_word)
        with st.expander("📊 Statistic in Numbers"):
            st.dataframe(word_count_stats.round(2))
    
    with col2:
        st.plotly_chart(fig_char)
        with st.expander("📊 Statistic in Numbers"):
            st.dataframe(char_count_stats.round(2))
    
    with st.expander("ℹ️ Interpretation"):
        st.markdown("""
        - Log scale helps visualize distribution across different magnitudes
        - Box plots show median, quartiles, and potential outliers
        - Helps compare text length characteristics between real and fake news
        """)