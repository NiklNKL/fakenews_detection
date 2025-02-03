import streamlit as st
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

def calibration_component(data):
    """Component for model calibration analysis"""
    
    if 'calibration_last_data' not in st.session_state:
        st.session_state.calibration_last_data = None
        st.session_state.calibration_fig = None
    
    st.markdown("""
        Analyze model calibration to understand how well predicted probabilities
        match actual outcomes.
    """)
    
    
    if st.session_state.calibration_last_data != data:
        fig = go.Figure()
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        for model, files in data.items():
            if "sklearn_predictions" in files:
                df = files["sklearn_predictions"]
                prob_true, prob_pred = calibration_curve(
                    df["true_label"], 
                    df["confidence_positive"], 
                    n_bins=10
                )
                fig.add_trace(go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    name=model,
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title="Calibration Plot",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.session_state.calibration_fig = fig
        st.session_state.calibration_last_data = data
    
    st.plotly_chart(st.session_state.calibration_fig, use_container_width=True)
    
   
    with st.expander("ℹ️ About Calibration"):
        st.markdown("""
            - Perfect calibration follows diagonal line
            - Above line: Model is underconfident
            - Below line: Model is overconfident
            - Well-calibrated models have:
                - Reliable probability estimates
                - Better uncertainty quantification
            - Important for:
                - Decision making
                - Risk assessment
                - Model reliability
        """)