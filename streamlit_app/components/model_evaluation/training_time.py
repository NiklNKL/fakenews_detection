import streamlit as st
import plotly.express as px

def format_time(seconds):
    """Convert seconds to formatted string (hours, minutes, seconds)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

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
        
        times_in_minutes = {model: time / 60 for model, time in times.items()}
        hover_texts = {model: format_time(time) for model, time in times.items()}
        
        fig = px.bar(
            x=list(times_in_minutes.keys()),
            y=list(times_in_minutes.values()),
            title="Training Time Comparison",
            labels={'x': 'Model', 'y': 'Total Training Time (minutes)'},
            text=list(hover_texts.values())
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
            
            System used: 
            - **CPU:** 13th Gen Intel(R) Core(TM) i5-13600K   3.50 GHz
            - **GPU:** NVIDIA GeForce RTX 3080
            - **RAM:** 32GB DDR5 5600MT/s
        """)