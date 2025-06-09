import os
import re
import pandas as pd
import glob
import time

def combine_cleaned_data():
    start_time = time.time()
    print("Starting cleaned data combination process...")
    
    # Output file path
    output_file = "combined_cleaned_synced_data.csv"
    
    # Define the header
    columns = [
        'dataset', 'num_trials', 'trial_number', 
        'time', 'channel_type', 'channel_name', 'value'
    ]
    
    # Create the CSV file with header
    with open(output_file, 'w', newline='') as f:
        f.write(','.join(columns) + '\n')
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir() if os.path.isdir(d) and d.startswith("Bicep_")]
    
    for dataset_dir in dataset_dirs:
        # Extract number of trials from directory name
        match = re.search(r'(\d+)_Trials', dataset_dir)
        if match:
            num_trials = match.group(1)
            print(f"Processing dataset: {dataset_dir} with {num_trials} trials")
            
            # Get all trial directories
            trial_dirs = glob.glob(os.path.join(dataset_dir, "trial_*"))
            
            for trial_dir in trial_dirs:
                # Extract trial number
                trial_match = re.search(r'trial_(\d+)', trial_dir)
                if trial_match:
                    trial_number = trial_match.group(1)
                    print(f"  Processing trial: {trial_number}")
                    
                    # Define file paths for cleaned data
                    eeg_file = os.path.join(trial_dir, "cleaned1_eeg_data.csv")
                    emg_file = os.path.join(trial_dir, "cleaned1_emg_data.csv")
                    
                    # Process cleaned EEG data if file exists
                    if os.path.exists(eeg_file):
                        print(f"    Processing cleaned EEG data...")
                        
                        # Process in chunks to save memory
                        chunk_size = 1000
                        chunk_iter = pd.read_csv(eeg_file, chunksize=chunk_size)
                        
                        for chunk in chunk_iter:
                            with open(output_file, 'a', newline='') as f:
                                for _, row in chunk.iterrows():
                                    time_point = row['time']
                                    # Get all channels (all columns except time)
                                    for col in chunk.columns:
                                        if col != 'time':
                                            # Write directly to file
                                            f.write(f"{dataset_dir},{num_trials},{trial_number},"
                                                  f"{time_point:.4f},EEG,{col},{row[col]:.6f}\n")
                        
                    # Process cleaned EMG data if file exists
                    if os.path.exists(emg_file):
                        print(f"    Processing cleaned EMG data...")
                        
                        # Process in chunks to save memory
                        chunk_size = 1000
                        chunk_iter = pd.read_csv(emg_file, chunksize=chunk_size)
                        
                        for chunk in chunk_iter:
                            with open(output_file, 'a', newline='') as f:
                                for _, row in chunk.iterrows():
                                    time_point = row['time']
                                    # Get all channels (all columns except time)
                                    for col in chunk.columns:
                                        if col != 'time':
                                            # Write directly to file
                                            f.write(f"{dataset_dir},{num_trials},{trial_number},"
                                                  f"{time_point:.4f},EMG,{col},{row[col]:.6f}\n")
    
    # Calculate file stats
    end_time = time.time()
    print(f"Complete! Process took {end_time - start_time:.2f} seconds")
    print(f"Combined cleaned data saved to {output_file}")
    
    # Count the number of rows in the output file
    row_count = 0
    with open(output_file, 'r') as f:
        for _ in f:
            row_count += 1
    
    # Subtract 1 for the header
    print(f"Total rows in combined dataset: {row_count - 1}")

if __name__ == "__main__":
    combine_cleaned_data()