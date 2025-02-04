import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def format_time(seconds):
    """Convert seconds to formatted string (hours, minutes, seconds)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def sklearn_model_performance_component(df, bert_data):
    """Component to visualize model performance metrics."""
    
    st.sidebar.markdown("### Options")
    include_bert = st.sidebar.toggle("Include BERT Models", value=False)
    if include_bert:
        metrics = ["accuracy", "f1", "precision", "recall"]
        models = {}
        for metric in metrics:
            values = {model: files["sklearn_metrics"][metric].iloc[0] 
                     for model, files in bert_data.items() 
                     if "sklearn_metrics" in files}

            sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
            models[metric] = sorted_values
            
        times = {model: files["summary_logs"]["train_runtime"].sum()
                for model, files in bert_data.items() 
                if "summary_logs" in files}

        models_set = set(model for values in models.values() for model, _ in values)
        rows = []
        for model in models_set:
            row = {"Model": model, "Dataset": "Test", "Accuracy": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan}
            for metric, values in models.items():
                for m, score in values:
                    if m == model:
                        row[metric.capitalize()] = score
            rows.append(row)
        for row in rows:
            row["Training_Time"] = times[row["Model"]]

        bert_values = pd.DataFrame(rows, columns=["Model", "Dataset", "Accuracy", "F1", "Precision", "Recall", "Training_Time"])

        df = pd.concat([df, bert_values], ignore_index=True)

    st.header("Sklearn Model Performance")
    st.write("All of the following non-BERT models used the TF-IDF vectorizer with a maximum of 10000 features.")

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    models = {}

    df_filtered = df[df["Dataset"] == "Test"]

    for metric in metrics:
        sorted_values = df_filtered.sort_values(by=metric, ascending=False)
        models[metric] = sorted_values["Model"].tolist()

    cols = st.columns(len(metrics))

    for metric, col in zip(metrics, cols):
        with col:
            st.markdown(f"### {metric} Score")
            table_data = []
            for rank, model in enumerate(models[metric], 1):
                train_score = df.loc[(df["Model"] == model) & (df["Dataset"] == "Train"), metric].values
                test_score = df.loc[(df["Model"] == model) & (df["Dataset"] == "Test"), metric].values[0]
                
                train_score = round(train_score[0],4) if len(train_score) > 0 else None
                overfitting = "❓" if train_score == None else ("⚠️" if train_score - test_score > 0.05 else "✅")
                
                table_data.append((rank, model, train_score, test_score, overfitting))
            
            df_table = pd.DataFrame(table_data, columns=["Place", "Model", "Train", "Test", "Overfitting"])
            st.table(df_table.drop(columns=["Place"]))

    with st.expander("ℹ️ About Performance Metrics"):
        st.markdown("""
            - **Accuracy**: Overall correctness of the model.
            - **Precision**: TP / (TP + FP), measures correctness of positive predictions.
            - **Recall**: TP / (TP + FN), measures completeness of positive predictions.
            - **F1-Score**: Harmonic mean of Precision and Recall.
        """)

    # Confusion Matrix Component
    st.markdown("""
        ## Confusion Matrices
        This component shows confusion matrices for each model, helping visualize 
        true positives, true negatives, false positives, and false negatives.
    """)

    if 'sklearn_confusion_matrix_last_data' not in st.session_state:
        st.session_state.sklearn_confusion_matrix_last_data = df
        st.session_state.sklearn_confusion_matrix_figs = {}

    for model in df["Model"].unique():
        model_df = df[(df["Model"] == model) & (df["Dataset"] == "Test")]
        if not model_df.empty and all(col in model_df.columns and pd.notna(model_df.iloc[0][col]) for col in ["tp", "tn", "fp", "fn"]):
            cm = [[model_df.iloc[0]["tn"], model_df.iloc[0]["fp"]],
                [model_df.iloc[0]["fn"], model_df.iloc[0]["tp"]]]
            
            cm_sum = sum(sum(row) for row in cm)
            cm_percentages = [[(val / cm_sum) * 100 for val in row] for row in cm] if cm_sum > 0 else [[0, 0], [0, 0]]
            
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
                title=model,
                height=400,
                width=400,
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label"
            )
            
            st.session_state.sklearn_confusion_matrix_figs[model] = fig

    st.session_state.sklearn_confusion_matrix_last_data = df

    for i in range(0, len(st.session_state.sklearn_confusion_matrix_figs), 3):
        cols = st.columns(min(3, len(st.session_state.sklearn_confusion_matrix_figs) - i))
        for (model, fig), col in zip(list(st.session_state.sklearn_confusion_matrix_figs.items())[i:i+3], cols):
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

    # Training Time Analysis
    st.markdown("""
        ## Training Time Analysis
        Compare the time taken by each model for training.
    """)

    df_time = df.pivot(index="Model", columns="Dataset", values="Training_Time")

    if 'sklearn_time_last_data' not in st.session_state:
        st.session_state.sklearn_time_last_data = None
        st.session_state.sklearn_time_fig = None

    
    times = {model: df.loc[df["Model"] == model, "Training_Time"].sum() for model in df["Model"].unique()}
    times_in_minutes = {model: time / 60 for model, time in times.items() if pd.notnull(time)}
    hover_texts = {model: format_time(time) for model, time in times.items() if pd.notnull(time)}
    
    fig = px.bar(
        x=list(times_in_minutes.keys()),
        y=list(times_in_minutes.values()),
        title="Training Time Comparison",
        labels={'x': 'Model', 'y': 'Total Training Time (minutes)'},
        text=list(hover_texts.values())
    )
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    
    st.session_state.sklearn_time_fig = fig
    st.session_state.sklearn_time_last_data = df

    st.plotly_chart(st.session_state.sklearn_time_fig, use_container_width=True)

    with st.expander("ℹ️ About Training Time"):
        st.markdown("""
            - Shows total training time for each model
            - Helps evaluate computational efficiency
            - Consider trade-off between time and performance
            - Useful for resource planning
        """)
        
