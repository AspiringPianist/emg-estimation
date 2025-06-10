# emg-estimation
## How to add new data? 
After collecting the data in a folder, place the folder containing all trials with appropriate naming format like `Bicep_{numtrials}_Trials_{Month}{Date}_{attempt}` in same directory with the `data/cleandata1.py` and `data/clean_all_datasets_in_folder.py`. After this, run `data/combine_all_cleaned_data.py` to get the CSV. For ease of use, the CSV is reformatted into a more understandable representation in `eeg_emg_analysis/eda_forsure.ipynb` which is what we use everywhere. 

If you don't want to clean data and instead observe the raw, unaligned data then just run `data/combine_all_data.csv` in the folder with all datasets (which each contain their trials). 



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
In the cleaned version, EMG is lagging behind EEG which aligns with the results of this [paper](https://iiitbac-my.sharepoint.com/:f:/g/personal/unnath_chittimalla_iiitb_ac_in/EoR21uJPk8pFgPrnIXmEQAoBTbSY4u_mL5qoUZqQK7UoFw?e=gYff6N). (symmetry.pdf in OneDrive)

`
From the spectrogram, it was confirmed that from 1 to 2 s before the motion started (when the
EMG signal rapidly arises), the power spectrum of the low-frequency band (lower than 7 Hz), the μ
rhythm (7–11 Hz) at the Fz (in the supplementary motor area), and C3 and C4 (in the primary motor
cortex area) increased. The moment after motion started (when flexing is beginning), the power of
the low-frequency band increased and reached the maximum level before the EMG signal amplitude
reached the maximum at Fz, C3, C4, CP1, and CP2 in the motor cortex. However, the power of the
low-frequency band at O1 and O2 (in the primary visual cortex) began to increase, and then reached
the maximum after the EMG signal amplitude reached the maximum level. Moreover, the power
spectrum of the μ rhythm in all channels began to decrease compared to that of the rest. At the moment
when the shoulder joint began to flex (when the EMG started to decrease), the power spectrum in
the low-frequency band increased again in all channels. After that, since the subject began to rest,
the power spectra of the μ rhythm increased, and the power spectra of the β waves (13–25 Hz) began
to decrease, as compared to during motion.
`

 
Use DPSS Taper window to remove noise in PSD of EEG. 

 
Training on features like std, mean, min, max, stft in windows using seq2seq produces the same EMG prediction (too much generalized). 

## Referenced Papers:
