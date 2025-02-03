import streamlit as st
import pandas as pd

# Define model aliases
model_aliases = {
    "roberta-base": "RoBERTa",
    "bert-base-uncased": "BERT",
    "distilbert-base-uncased": "DistilBERT"
}

def metrics_comparison_component(data):
    """Component for displaying model metrics comparison as tables inside Streamlit columns"""
    
    if 'metrics_last_data' not in st.session_state:
        st.session_state.metrics_last_data = None
        st.session_state.metrics_values = {}
    
    metrics = ["accuracy", "f1", "precision", "recall"]
    models = {}
    
    if st.session_state.metrics_last_data != data:
        for metric in metrics:
            values = {model: files["sklearn_metrics"][metric].iloc[0] 
                     for model, files in data.items() 
                     if "sklearn_metrics" in files}

            sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
            models[metric] = sorted_values
        
        st.session_state.metrics_values = models
        st.session_state.metrics_last_data = data

    cols = st.columns(len(metrics) + 1)

    for i, (metric, col) in enumerate(zip(metrics, cols[:-1])):
        with col:
            st.markdown(f"### {metric.capitalize()} Score")
            table_data = []
            for rank, (model, score) in enumerate(st.session_state.metrics_values[metric], 1):
                base_model_name = model.split(" ")[0] 
                model_name = model_aliases.get(base_model_name, base_model_name)
                
                peft_status = "Yes" if "(with_peft)" in model else "No"
                
                table_data.append((rank, model_name, score, peft_status))
            
            df = pd.DataFrame(table_data, columns=["Place", "Model", metric.capitalize(), "PEFT"])
            st.markdown("""
                <style>
                    .stTable {
                        border-spacing: 10px;
                    }
                    .stTable th {
                        background-color: #f5f5f5;
                        color: #333;
                        padding: 12px;
                    }
                    .stTable td {
                        padding: 12px;
                    }
                    .stTable td:nth-child(4) {
                        text-align: center;
                    }
                    .stTable td[style*="Yes"] {
                        background-color: lightgreen;
                    }
                    .stTable td[style*="No"] {
                        background-color: coral;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.table(df.drop(columns=["Place"]))
