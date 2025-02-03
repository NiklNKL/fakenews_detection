import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
import plotly.express as px

def confidence_distribution_component(data):
    """Component for displaying prediction confidence distributions using KDE curves"""
    
    if 'conf_dist_last_data' not in st.session_state:
        st.session_state.conf_dist_last_data = None
        st.session_state.conf_dist_fig = None
    
    st.markdown("""
        Analyze the distribution of model prediction confidences.
        This helps understand how certain the model is about its predictions.
    """)
    

    if st.session_state.conf_dist_last_data != data:
        fig = go.Figure()
        
        for model, files in data.items():
            if "sklearn_predictions" in files:
                df = files["sklearn_predictions"]
                # Calculate KDE values
                kde = gaussian_kde(df["confidence_positive"])
                x_range = np.linspace(0, 1, 200)
                y_values = kde(x_range)
                
                # Add KDE curve
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_values,
                    name=model,
                    mode='lines',
                    fill='tozeroy',  # Fill area under curve
                    fillcolor=px.colors.qualitative.Set3[list(data.keys()).index(model) % len(px.colors.qualitative.Set3)],
                    line=dict(width=2),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Prediction Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Density",
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            xaxis=dict(
                range=[0, 1],
                tickformat='.1f'
            )
        )
        
        st.session_state.conf_dist_fig = fig
        st.session_state.conf_dist_last_data = data
    
    st.plotly_chart(st.session_state.conf_dist_fig, use_container_width=True)
    
    
    with st.expander("ℹ️ About Confidence Distribution"):
        st.markdown("""
            - Shows how confident the model is in its predictions
            - Smooth curves represent the density of predictions at each confidence level
            - Peaks indicate common confidence values
            - Ideal distribution often shows clear separation between classes
            - Compare patterns across different models:
                - Sharp peaks suggest high certainty
                - Broad distributions suggest uncertainty
                - Multiple peaks may indicate distinct prediction patterns
        """)