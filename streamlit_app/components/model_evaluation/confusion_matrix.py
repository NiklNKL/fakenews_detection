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
    
    model_aliases = {
        "roberta-base": "RoBERTa",
        "bert-base-uncased": "BERT",
        "distilbert-base-uncased": "DistilBERT"
    }
    
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
                
                base_model_name = model.split(" ")[0] 
                model_name = model_aliases.get(base_model_name, base_model_name)
                if "(with_peft)" in model:
                    model_name += " with PEFT"
                else:
                    model_name += " without PEFT"
                fig.update_layout(
                    title=model_name,
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