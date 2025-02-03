import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve


def curves_component(data):
    """Component for displaying ROC and PR curves"""
    
    if 'curves_last_data' not in st.session_state:
        st.session_state.curves_last_data = None
        st.session_state.curves_fig = None
   
    if st.session_state.curves_last_data != data:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ROC Curve", "Precision-Recall Curve")
        )
        
        for model, files in data.items():
            if "sklearn_roc" in files:
                roc_df = files["sklearn_roc"]
                fig.add_trace(
                    go.Scatter(
                        x=roc_df["false_positive_rate"],
                        y=roc_df["true_positive_rate"],
                        name=f"{model} (AUC: {roc_df['auc'].iloc[0]:.3f})",
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            if "sklearn_predictions" in files:
                df = files["sklearn_predictions"]
                precision, recall, _ = precision_recall_curve(
                    df["true_label"], df["confidence_positive"]
                )
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision,
                        name=model,
                        mode='lines'
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(height=500)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        st.session_state.curves_fig = fig
        st.session_state.curves_last_data = data
    st.markdown("##### Model Performance Curves")
    st.markdown("""
    Visualize model performance through ROC and Precision-Recall curves.
    These curves help understand the trade-offs in model predictions.
    """)
    st.plotly_chart(st.session_state.curves_fig, use_container_width=True)
    
    
    with st.expander("ℹ️ About These Curves"):
        st.markdown("""
            **ROC Curve**:
            - Shows trade-off between true and false positive rates
            - AUC closer to 1.0 indicates better performance
            
            **Precision-Recall Curve**:
            - Shows trade-off between precision and recall
            - Useful for imbalanced datasets
            - Higher curve indicates better performance
        """)
