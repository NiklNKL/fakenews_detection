import streamlit as st
import plotly.graph_objects as go

def learning_rate_analysis_component(data):
    """Component for analyzing learning rate behavior"""
    
    if 'lr_analysis_last_data' not in st.session_state:
        st.session_state.lr_analysis_last_data = None
        st.session_state.lr_figs = {}
    
    st.markdown("""
        Analyze learning rate behavior and its effects during training.
        This helps understand the optimization process and learning dynamics.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.lr_analysis_last_data != data:
            # Learning Rate vs Loss
            lr_loss_fig = go.Figure()
            for model, files in data.items():
                if "training_logs" in files:
                    df = files["training_logs"]
                    lr_loss_fig.add_trace(go.Scatter(
                        x=df["learning_rate"],
                        y=df["loss"],
                        name=model,
                        mode='lines'
                    ))
            
            lr_loss_fig.update_layout(
                title="Learning Rate vs Loss",
                xaxis_title="Learning Rate",
                yaxis_title="Loss",
                height=400
            )
            st.session_state.lr_figs['lr_loss'] = lr_loss_fig
        
        st.plotly_chart(st.session_state.lr_figs['lr_loss'], use_container_width=True)
    
    with col2:
        if st.session_state.lr_analysis_last_data != data:
            # Learning Rate over Steps
            lr_steps_fig = go.Figure()
            for model, files in data.items():
                if "training_logs" in files:
                    df = files["training_logs"]
                    lr_steps_fig.add_trace(go.Scatter(
                        x=df["step"],
                        y=df["learning_rate"],
                        name=model,
                        mode='lines'
                    ))
            
            lr_steps_fig.update_layout(
                title="Learning Rate over Steps",
                xaxis_title="Step",
                yaxis_title="Learning Rate",
                height=400
            )
            st.session_state.lr_figs['lr_steps'] = lr_steps_fig
            st.session_state.lr_analysis_last_data = data
        
        st.plotly_chart(st.session_state.lr_figs['lr_steps'], use_container_width=True)
    
    with st.expander("ℹ️ About Learning Rate Analysis"):
        st.markdown("""
            - **Learning Rate vs Loss**: Shows how loss changes with different learning rates
            - **Learning Rate over Steps**: Shows the learning rate schedule during training
            - Important for understanding:
                - Learning rate scheduling effectiveness
                - Optimal learning rate ranges
                - Training stability
        """)