import streamlit as st
import plotly.express as px


def metrics_comparison_component(data):
    """Component for displaying model metrics comparison"""
    
    if 'metrics_last_data' not in st.session_state:
        st.session_state.metrics_last_data = None
        st.session_state.metrics_figs = {}
    
    st.markdown("""
        Compare different evaluation metrics across models.
        This helps identify which models perform best on different criteria.
    """)
    
    metrics = ["accuracy", "f1", "precision"]
    cols = st.columns(len(metrics))
    
    if st.session_state.metrics_last_data != data:
        for metric in metrics:
            values = {model: files["sklearn_metrics"][metric].iloc[0] 
                     for model, files in data.items() 
                     if "sklearn_metrics" in files}
            
            fig = px.bar(
                x=list(values.keys()),
                y=list(values.values()),
                title=f"{metric.capitalize()} Score",
                labels={'x': 'Model', 'y': metric.capitalize()}
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            st.session_state.metrics_figs[metric] = fig
        
        st.session_state.metrics_last_data = data
    
    for metric, col in zip(metrics, cols):
        with col:
            st.plotly_chart(st.session_state.metrics_figs[metric], use_container_width=True)