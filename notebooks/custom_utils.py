import os
import pandas as pd
import math
import joblib
import sys

_path_added = False

def add_project_root_to_path():
    """
    Check if the project root directory is in the Python path.
    If not, add it to sys.path and change the working directory to the project root.
    """
    global _path_added

    if _path_added:
        return
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    if root_dir not in sys.path:
        sys.path.append(root_dir)
        # print(f"Added {root_dir} to Python path.")
    
    if os.getcwd() != root_dir:
        os.chdir(root_dir)
        # print(f"Changed working directory to {root_dir}.")
    _path_added = True

def load_and_concatenate_parquet_files(folder_path):
    add_project_root_to_path()
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    files.sort()
    df_list = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    return concatenated_df

def save_dataframe_as_parquet(df:pd.DataFrame, folder_path="data", folder_name=None, always_overwrite=None, model_object=None):
    add_project_root_to_path()
    if not folder_name:
        folder_name = input("Enter the name of the folder to save the files: ")

    full_path = os.path.join(folder_path, folder_name)

    # Check if the folder already exists
    if os.path.exists(full_path) and always_overwrite is not True:
        if always_overwrite is None:
            overwrite = input(f"The folder '{folder_name}' already exists. Do you want to overwrite it? (yes/no): ")
            always_overwrite = overwrite.lower() != 'yes'
        if not always_overwrite:
            suffix = 1
            new_folder_name = f"{folder_name}_{suffix}"
            while os.path.exists(os.path.join(folder_path, new_folder_name)):
                suffix += 1
                new_folder_name = f"{folder_name}_{suffix}"
            folder_name = new_folder_name
            full_path = os.path.join(folder_path, folder_name)
    
    os.makedirs(full_path, exist_ok=True)

    temp_file = os.path.join(full_path, "temp.parquet")
    df.to_parquet(temp_file)
    file_size = os.path.getsize(temp_file) / (1024 * 1024)  # Size in MB
    os.remove(temp_file)

    if file_size > 50:
        num_splits = math.ceil(file_size / 50)
        row_split = math.ceil(len(df) / num_splits)
    else:
        num_splits = 1
        row_split = len(df)
    
    for i in range(num_splits):
        start_row = i * row_split
        end_row = min((i + 1) * row_split, len(df))
        split_df = df.iloc[start_row:end_row]
        split_file_name = os.path.join(full_path, f"{folder_name}_part_{i + 1}.parquet")
        split_df.to_parquet(split_file_name)
    
    if model_object:
        for key, value in model_object.items():
            joblib.dump(value, f'{full_path}/{key}_model.pkl')

    print(f"Dataframe saved in {num_splits} files under the folder: {full_path}")
    
    return full_path