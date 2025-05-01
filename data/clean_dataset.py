from clean_data import clean_and_save_data
import pandas as pd
import os

data_root = input("Enter the path to the data root directory: ")
trials = input("Enter the number of trials: ")
trials = int(trials)
if not os.path.exists(data_root):
    raise ValueError(f"Data root directory {data_root} does not exist.")

for i in range(1, trials):
    print(f"Processing Trial {i}...\n\n")
    try:
        eeg_data = pd.read_csv(os.path.join(data_root, f'Trial_{i:02d}/eeg_data.csv'))
        emg_data = pd.read_csv(os.path.join(data_root, f'Trial_{i:02d}/emg_data.csv'))
        clean_and_save_data(eeg_data, emg_data, os.path.join(data_root, f'Trial_{i:02d}/cleaned_eeg_data.csv'), os.path.join(data_root, f'Trial_{i:02d}/cleaned_emg_data.csv'), os.path.join(data_root, f'Trial_{i:02d}/ica_components.csv'))
    except Exception as e:
        print(f"Error processing Trial {i}: {e}")
    print(f"\n\n")