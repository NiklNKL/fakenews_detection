import streamlit as st
import plotly.graph_objects as go

def training_metrics_component(data):
    """Component for tracking various training metrics over time"""
    
    if 'training_metrics_last_data' not in st.session_state:
        st.session_state.training_metrics_last_data = None
        st.session_state.training_metrics_figs = {}
    
    st.markdown("""
        Track various training metrics over time to understand model behavior
        and performance during training.
    """)
    
    # Create tabs for different metric groups
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Training Dynamics", "Speed Metrics"])
    
    if st.session_state.training_metrics_last_data != data:
        # Performance Metrics Over Time
        metrics_fig = go.Figure()
        for model, files in data.items():
            if "eval_logs" in files:
                df = files["eval_logs"]
                for metric in ['accuracy', 'f1', 'precision', 'recall']:
                    if metric in df.columns:
                        metrics_fig.add_trace(go.Scatter(
                            x=df["step"],
                            y=df[metric],
                            name=f"{model} - {metric}",
                            line=dict(dash='solid' if metric == 'accuracy' else 'dash')
                        ))
        
        metrics_fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Step",
            yaxis_title="Score",
            height=500
        )
        st.session_state.training_metrics_figs['performance'] = metrics_fig
        
        # Gradient Norm Over Time
        grad_fig = go.Figure()
        for model, files in data.items():
            if "training_logs" in files:
                df = files["training_logs"]
                if "grad_norm" in df.columns:
                    grad_fig.add_trace(go.Scatter(
                        x=df["step"],
                        y=df["grad_norm"],
                        name=model
                    ))
        
        grad_fig.update_layout(
            title="Gradient Norm Over Time",
            xaxis_title="Step",
            yaxis_title="Gradient Norm",
            height=500
        )
        st.session_state.training_metrics_figs['gradients'] = grad_fig
        
        # Training Speed
        if any("training_logs" in files for model, files in data.items()):
            speed_fig = go.Figure()
            speed_stats = {}
            
            for model, files in data.items():
                if "training_logs" in files:
                    df = files["eval_logs"]
                    if "samples_per_second" in df.columns:
                        speeds = df["samples_per_second"]
                        speed_stats[model] = {
                            'mean': speeds.mean(),
                            'min': speeds.min(),
                            'max': speeds.max()
                        }
                        
                        speed_fig.add_trace(go.Box(
                            y=speeds,
                            name=model,
                            boxpoints='outliers'
                        ))
            
            speed_fig.update_layout(
                title="Training Speed Distribution",
                yaxis_title="Samples per Second",
                height=500
            )
            st.session_state.training_metrics_figs['speed'] = speed_fig
            st.session_state.training_metrics_figs['speed_stats'] = speed_stats
        
        st.session_state.training_metrics_last_data = data
    
    # Display metrics in tabs
    with tab1:
        st.plotly_chart(st.session_state.training_metrics_figs['performance'], use_container_width=True)
    
    with tab2:
        st.plotly_chart(st.session_state.training_metrics_figs['gradients'], use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(st.session_state.training_metrics_figs['speed'], use_container_width=True)
        with col2:
            if 'speed_stats' in st.session_state.training_metrics_figs:
                st.markdown("### Speed Statistics")
                for model, stats in st.session_state.training_metrics_figs['speed_stats'].items():
                    st.markdown(f"""
                        **{model}**:
                        - Mean: {stats['mean']:.2f} samples/s
                        - Min: {stats['min']:.2f} samples/s
                        - Max: {stats['max']:.2f} samples/s
                    """)