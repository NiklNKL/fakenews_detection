import streamlit as st
import plotly.graph_objects as go

def confusion_matrix_component(data):
    """Component for displaying confusion matrices for each model"""
    
    if 'confusion_matrix_last_data' not in st.session_state:
        st.session_state.confusion_matrix_last_data = None
        st.session_state.confusion_matrix_figs = {}
    
    st.markdown("""
        This component shows confusion matrices for each model, helping visualize 
        true positives, true negatives, false positives, and false negatives.
    """)
    
   # Create a list of models with confusion matrices
    model_names = [model for model, files in data.items() if "sklearn_confusion" in files]
    
    if st.session_state.confusion_matrix_last_data != data:
        for model, files in data.items():
            if "sklearn_confusion" in files:
                df = files["sklearn_confusion"]
                cm = df.iloc[:, 1:].values.reshape(2, 2)
                
                # Calculate percentages for annotations
                cm_sum = cm.sum()
                cm_percentages = cm / cm_sum * 100
                
                # Create annotations with both count and percentage
                annotations = [
                    [f"{int(v)}<br>({p:.1f}%)" for v, p in zip(row, prow)]
                    for row, prow in zip(cm, cm_percentages)
                ]
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    text=annotations,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorscale="Blues",
                    showscale=False
                ))
                
                fig.update_layout(
                    title=f"Confusion Matrix - {model}",
                    height=400,
                    width=400,
                    xaxis_title="Predicted Label",
                    yaxis_title="Actual Label"
                )
                
                st.session_state.confusion_matrix_figs[model] = fig
        
        st.session_state.confusion_matrix_last_data = data
    
    # Display matrices in a grid (max 3 per row)
    for i in range(0, len(st.session_state.confusion_matrix_figs), 3):
        cols = st.columns(min(3, len(st.session_state.confusion_matrix_figs) - i))
        for (model, fig), col in zip(list(st.session_state.confusion_matrix_figs.items())[i:i+3], cols):
            with col:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add metrics derived from confusion matrix
                if "sklearn_confusion" in data[model]:
                    df = data[model]["sklearn_confusion"]
                    cm = df.iloc[:, 1:].values.reshape(2, 2)
                    tn, fp, fn, tp = cm.ravel()
                    
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    with st.expander("📊 Matrix Metrics"):
                        st.markdown(f"""
                            - **Accuracy**: {accuracy:.3f}
                            - **Precision**: {precision:.3f}
                            - **Recall**: {recall:.3f}
                            - **F1 Score**: {f1:.3f}
                        """)
    
    with st.expander("ℹ️ About Confusion Matrix"):
        st.markdown("""
            The confusion matrix shows:
            
            - **True Negatives** (top-left): Correctly predicted negatives
            - **False Positives** (top-right): Incorrectly predicted positives
            - **False Negatives** (bottom-left): Incorrectly predicted negatives
            - **True Positives** (bottom-right): Correctly predicted positives
            
            Each cell shows:
            - The absolute count
            - The percentage of total predictions (in parentheses)
            
            💡 A good model should have high numbers in the diagonal (top-left to bottom-right)
            and low numbers in the off-diagonal.
        """)