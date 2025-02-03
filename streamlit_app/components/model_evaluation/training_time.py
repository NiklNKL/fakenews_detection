import streamlit as st
import plotly.express as px

def training_time_component(data):
    """Component for displaying training time comparison"""
    
    if 'time_last_data' not in st.session_state:
        st.session_state.time_last_data = None
        st.session_state.time_fig = None
    
    st.markdown("""
        Compare training time across different models.
        This helps understand the computational efficiency of each approach.
    """)
    
    
    if st.session_state.time_last_data != data:
        times = {model: files["summary_logs"]["train_runtime"].sum()
                for model, files in data.items() 
                if "summary_logs" in files}
        
        fig = px.bar(
            x=list(times.keys()),
            y=list(times.values()),
            title="Training Time Comparison",
            labels={'x': 'Model', 'y': 'Total Training Time (s)'}
        )
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45
        )
        
        st.session_state.time_fig = fig
        st.session_state.time_last_data = data
    
    st.plotly_chart(st.session_state.time_fig, use_container_width=True)
    
    
    with st.expander("ℹ️ About Training Time"):
        st.markdown("""
            - Shows total training time for each model
            - Helps evaluate computational efficiency
            - Consider trade-off between time and performance
            - Useful for resource planning
        """)