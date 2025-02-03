import streamlit as st
import plotly.graph_objects as go


def loss_plots_component(data):
    """Component for displaying training and evaluation loss plots"""
    
    if 'loss_plots_last_data' not in st.session_state:
        st.session_state.loss_plots_last_data = None
        st.session_state.loss_fig = None
    
    st.markdown("""
        This component shows the training and evaluation loss curves over time.
        It helps visualize model convergence and potential overfitting.
    """)
    

    # Create figure if data has changed
    if st.session_state.loss_plots_last_data != data:
        fig = go.Figure()
        
        for model, files in data.items():
            if "training_logs" in files:
                df = files["training_logs"]
                fig.add_trace(go.Scatter(
                    x=df["step"],
                    y=df["loss"],
                    name=f"{model} (Train)",
                    mode='lines'
                ))
            if "eval_logs" in files:
                df = files["eval_logs"]
                fig.add_trace(go.Scatter(
                    x=df["step"],
                    y=df["loss"],
                    name=f"{model} (Eval)",
                    line=dict(dash='dash')
                ))
        
        fig.update_layout(
            title="Training and Evaluation Loss",
            xaxis_title="Step",
            yaxis_title="Loss",
            height=500
        )
        
        st.session_state.loss_fig = fig
        st.session_state.loss_plots_last_data = data
    
    st.plotly_chart(st.session_state.loss_fig, use_container_width=True)
    
    
    with st.expander("ℹ️ About Loss Plots"):
        st.markdown("""
            - **Training Loss**: Shows how well the model is fitting the training data
            - **Evaluation Loss**: Shows model performance on validation data
            - **Gap Analysis**: Large gap between curves may indicate overfitting
            - **Convergence**: Flattening curves suggest model convergence
        """)