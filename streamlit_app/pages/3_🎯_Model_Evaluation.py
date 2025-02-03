import streamlit as st
import os
import pandas as pd
import glob
from pathlib import Path
from components.evaluation_components import (
    loss_plots_component,
    metrics_comparison_component,
    curves_component,
    confidence_distribution_component,
    training_time_component,
    confusion_matrix_component,
    calibration_component,
    training_metrics_component,
    learning_rate_analysis_component
)

@st.cache_resource
def load_model_results(base_path):
    data = {}
    for peft_type in ["with_peft", "without_peft"]:
        peft_path = os.path.join(base_path, peft_type)
        if not os.path.exists(peft_path):
            continue
        
        for model_name in os.listdir(peft_path):
            model_path = os.path.join(peft_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            model_key = f"{model_name} ({peft_type})"
            data[model_key] = {}
            
            for file in glob.glob(os.path.join(model_path, "*.parquet")):
                file_key = os.path.basename(file).replace(".parquet", "")
                data[model_key][file_key] = pd.read_parquet(file)
    
    return data

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🎯",
    layout="wide"
)

st.title("Model Analysis Dashboard")
st.write("Interactive analysis of model training and evaluation results")

root_path = Path(__file__).resolve().parent.parent.parent 

path = f"{root_path}/data/model_evaluation"

data = load_model_results(path)

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Analysis Section",
    ["Training Progress", "Model Performance"]
)

if section == "Training Progress":
    loss_plots_component(data)
    training_time_component(data)
    training_metrics_component(data)
    learning_rate_analysis_component(data)
    
elif section == "Model Performance":
    metrics_comparison_component(data)
    confusion_matrix_component(data)
    curves_component(data)
    confidence_distribution_component(data)
    calibration_component(data)
    
    
        


    
