from clean_data1 import clean_and_save_data
import pandas as pd
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cleaning_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def is_dataset_folder(path):
    """Check if folder contains required dataset files"""
    return os.path.isfile(os.path.join(path, 'eeg_data.csv')) and \
           os.path.isfile(os.path.join(path, 'emg_data.csv'))

def find_dataset_folders(root_dir):
    """Find all folders containing EEG/EMG datasets"""
    dataset_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if is_dataset_folder(dirpath):
            dataset_folders.append(dirpath)
            
    return dataset_folders

def process_dataset(dataset_path):
    """Process a single dataset folder"""
    try:
        # Load data
        eeg_data = pd.read_csv(os.path.join(dataset_path, 'eeg_data.csv'))
        emg_data = pd.read_csv(os.path.join(dataset_path, 'emg_data.csv'))
        
        # Create output paths
        cleaned_eeg_path = os.path.join(dataset_path, 'cleaned1_eeg_data.csv')
        cleaned_emg_path = os.path.join(dataset_path, 'cleaned1_emg_data.csv')
        
        # Process data
        clean_and_save_data(eeg_data, emg_data, cleaned_eeg_path, cleaned_emg_path)
        logging.info(f"Successfully processed dataset in {dataset_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing dataset in {dataset_path}: {str(e)}")
        return False

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting dataset cleaning process")
    
    # Get root directory
    root_dir = input("Enter the path to the root directory containing datasets: ")
    if not os.path.exists(root_dir):
        logging.error(f"Root directory {root_dir} does not exist")
        return
    
    # Find all dataset folders
    dataset_folders = find_dataset_folders(root_dir)
    logging.info(f"Found {len(dataset_folders)} dataset folders")
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for folder in dataset_folders:
        if process_dataset(folder):
            successful += 1
        else:
            failed += 1
    
    # Log summary
    logging.info("\nProcessing Summary:")
    logging.info(f"Total datasets found: {len(dataset_folders)}")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed to process: {failed}")
    logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main()
