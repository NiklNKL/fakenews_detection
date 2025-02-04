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
    learning_rate_analysis_component,
    model_performance_info_component,
    sklearn_model_performance_component,
)
from components.utils import model_selection_component
root_path = Path(__file__).resolve().parent.parent.parent 

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

@st.cache_data
def load_sklearn_model_logs():
    return pd.read_parquet(f"{root_path}/data/model_evaluation/sklearn_models_evaluation.parquet")


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🎯",
    layout="wide"
)

st.title("Model Analysis Dashboard")
st.write("Interactive analysis of model training and evaluation results")

path = f"{root_path}/data/model_evaluation"

data = load_model_results(path)
sklearn_logs = load_sklearn_model_logs()

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Analysis Section",
    ["BERT Training Progress", "BERT Model Performance", "Traditional Models"]
)


if section == "BERT Training Progress":
    st.sidebar.title("Model Selection")
    filtered_data = model_selection_component(data)
    loss_plots_component(filtered_data)
    training_time_component(filtered_data)
    training_metrics_component(filtered_data)
    learning_rate_analysis_component(filtered_data)
    
elif section == "BERT Model Performance":
    st.sidebar.title("Model Selection")
    filtered_data = model_selection_component(data)
    model_performance_info_component()
    metrics_comparison_component(data)
    confusion_matrix_component(data)
    curves_component(filtered_data)
    confidence_distribution_component(filtered_data)
    calibration_component(filtered_data)

elif section == "Traditional Models":
    sklearn_model_performance_component(sklearn_logs, data)
    
        


    
