# emg-estimation
## Data
Link to updated dataset (as of June 10th, 2025): [OneDrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/unnath_chittimalla_iiitb_ac_in/EoR21uJPk8pFgPrnIXmEQAoBTbSY4u_mL5qoUZqQK7UoFw?e=gYff6N)

## Exploratory Data Analysis 
Done in `eeg_emg_analysis/eda_forsure.ipynb` and also checkout `eeg_emg_analysis/stft_eeg_vs_emg.py`

## Labelling/Filter Tool
Check `data/labelling.py`
![image](https://github.com/user-attachments/assets/0a268852-b392-4b49-a3dc-47ab7c041ca5)


## Current Status
In the cleaned version, EEG is lagging behind EMG which shouldn't happen. 

 
Use DPSS Taper window to remove noise in PSD of EEG. 

 
Training on features like std, mean, min, max, stft in windows using seq2seq produces the same EMG prediction (too much generalized). 
