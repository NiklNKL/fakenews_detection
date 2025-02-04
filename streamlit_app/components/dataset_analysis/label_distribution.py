import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def create_pie_chart(data, title, show_percents=True):
    """
    Helper function to create a pie chart with percentages and counts
    """
    # Calculate total for percentage computation
    total = sum(data['count'])
    
    # Create labels with both count and percentage
    labels = [f"{label}<br>{count:,} ({(count/total)*100:.1f}%)" 
             for label, count in zip(data['label_names'], data['count'])]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=data['count'],
        textposition='inside',
        textinfo='label',
        showlegend=False,
        marker=dict(colors=['#4ECB71', '#FF6B6B'])  # Green for real, Red for fake
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=400
    )
    
    return fig

def dataset_distribution_component(data):
    """
    Component for visualizing dataset distributions with pie charts
    """
    st.header("Dataset Distribution Analysis")
    
    # Create columns for all pie charts
    col1, col2, col3, col4 = st.columns(4)
    
    # Original dataset
    original_data = data[data['dataset'] == 'original']
    fig_original = create_pie_chart(original_data, "Original Dataset")
    
    with col1:
        st.plotly_chart(fig_original, use_container_width=True)
    
    # Training splits
    splits = ['train', 'valid', 'test']
    cols = [col2, col3, col4]
    
    for split, col in zip(splits, cols):
        split_data = data[data['dataset'] == split]
        fig = create_pie_chart(split_data, f"{split.title()} Set")
        
        with col:
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics and Interpretation in two columns below the charts
    col_stats, col_interp = st.columns(2)
    
    with col_stats:
        with st.expander("📊 Dataset Statistics"):
            st.subheader("Original Dataset")
            st.dataframe(
                original_data[['label_names', 'count']]
                .set_index('label_names')
                .round(4)
            )
            
            st.subheader("Split Statistics")
            split_stats = data[data['dataset'] != 'original']
            split_stats_pivot = split_stats.pivot(
                index='dataset', 
                columns='label_names', 
                values=['count']
            ).round(4)
            st.dataframe(split_stats_pivot)
    
    with col_interp:
        with st.expander("ℹ️ Dataset Analysis"):
            total = original_data['count'].sum()
            real_pct = original_data[original_data['label_names'] == 'real']['proportion'].iloc[0]*100
            fake_pct = original_data[original_data['label_names'] == 'fake']['proportion'].iloc[0]*100
            
            st.subheader("Original Dataset Composition")
            st.markdown(f"""
            - Total samples: {total:,}
            - The dataset shows a relatively balanced distribution:
                - Real news: {real_pct:.1%} of the dataset
                - Fake news: {fake_pct:.1%} of the dataset
            - This balance is beneficial for model training as it reduces potential bias
            """)
            
            st.subheader("Split Analysis")
            train_total = data[data['dataset'] == 'train']['count'].sum()
            valid_total = data[data['dataset'] == 'valid']['count'].sum()
            test_total = data[data['dataset'] == 'test']['count'].sum()
            
            st.markdown(f"""
            **Sample Distribution:**
            - Training set: {train_total:,} samples ({train_total/total:.1%})
            - Validation set: {valid_total:,} samples ({valid_total/total:.1%})
            - Test set: {test_total:,} samples ({test_total/total:.1%})
            
            **Key Observations:**
            - All splits maintain similar class distribution ratios
            - The training set contains the majority of the data
            - Validation and test sets are appropriately sized for model evaluation
            
            **Class Balance:**
            - Each split maintains approximately 55/45 ratio between fake and real news
            - This consistent distribution helps ensure:
                - Reliable model training
                - Representative validation results
                - Accurate test set evaluation
            """)