# emg-estimation
## How to add new data? 
After collecting the data in a folder, place the folder containing all trials with appropriate naming format like `Bicep_{numtrials}_Trials_{Month}{Date}_{attempt}` in same directory with the `cleandata1.py` and `clean_all_datasets_in_folder.py`. After this, run `combine_all_cleaned_data.py` to get the CSV. For ease of use, the CSV is reformatted into a more understandable representation in `eeg_emg_analysis/eda_forsure.ipynb` which is what we use everywhere. 


## Data
Link to updated dataset (as of June 10th, 2025): [OneDrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/unnath_chittimalla_iiitb_ac_in/EoR21uJPk8pFgPrnIXmEQAoBTbSY4u_mL5qoUZqQK7UoFw?e=gYff6N) 

 Dominant EEG Channels - `O2, C3 and Cz` (according to Common Referential 19 Montage)

Respective Channel Numbers - `16, 9, 18`

## Exploratory Data Analysis 
Done in `eeg_emg_analysis/eda_forsure.ipynb` and also checkout `eeg_emg_analysis/stft_eeg_vs_emg.py`

## Labelling/Filter Tool
Check `data/labelling.py`
![image](https://github.com/user-attachments/assets/0a268852-b392-4b49-a3dc-47ab7c041ca5)


## Current Status
In the cleaned version, EEG is lagging behind EMG which shouldn't happen. 

 
Use DPSS Taper window to remove noise in PSD of EEG. 

 
Training on features like std, mean, min, max, stft in windows using seq2seq produces the same EMG prediction (too much generalized). 
