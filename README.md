# emg-estimation
Model Training Plots

<img width="4470" height="2966" alt="image" src="https://github.com/user-attachments/assets/ba15ce2a-93af-4414-a988-4327550a6a7b" />
<img width="2970" height="1765" alt="image" src="https://github.com/user-attachments/assets/d18b3a8c-2080-40bd-9250-0fd4c3d5ce74" />


## How to add new data? 
After collecting the data in a folder, place the folder containing all trials with appropriate naming format like `Anything_{numtrials}_Trials_{Month}{Date}_{attempt}` in same directory with the `data/cleandata2.py` and `data/clean_all_datasets_in_folder.py`. After this, run `data/arrange_all_cleaned_data.py` to get the CSVs into a directory like `./cleaned_data`.

## Data
Link to updated dataset (as of June 20th, 2025): [OneDrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/unnath_chittimalla_iiitb_ac_in/EoR21uJPk8pFgPrnIXmEQAoBTbSY4u_mL5qoUZqQK7UoFw?e=gYff6N) 

## Exploratory Data Analysis 
Basic data analysis done in `eeg_emg_analysis/eda_forsure.ipynb` and also checkout `eeg_emg_analysis/stft_eeg_vs_emg.py`.

## Labelling/Filter Tool
Check `data/labelling.py`
![image](https://github.com/user-attachments/assets/0a268852-b392-4b49-a3dc-47ab7c041ca5)

## Current Status
Test R2-score : 0.23292342004137473 
Correlation score: 0.7837367363450538 
