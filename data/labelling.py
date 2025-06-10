import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.signal import butter, filtfilt
import uuid
import re

# --- Bandpass Filter Functions ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    """Designs a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=4):
    """Applies the bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Use filtfilt for zero-phase filtering
    filtered = filtfilt(b, a, data)
    return filtered

class EEGLabeler:
    """A GUI tool for labeling EEG and EMG time-series data."""

    def __init__(self, master):
        self.master = master
        self.master.title("EEG/EMG Time-Series Labeling Tool")
        self.master.geometry("1400x900")

        # --- Data Attributes ---
        self.df = None
        self.current_trial_data = None
        self.eeg_channels = []
        self.emg_channels = []
        self.labeled_spans = {}
        self.calculated_sfreq = None

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Sidebar & Plot Panel ---
        self.sidebar = ttk.Frame(self.main_frame, width=350)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar.pack_propagate(False)

        self.plot_panel = ttk.Frame(self.main_frame)
        self.plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_sidebar_widgets()
        self._create_plot_widgets()

    def _create_sidebar_widgets(self):
        """Creates all control widgets in the left sidebar."""
        # --- File Loading ---
        ttk.Button(self.sidebar, text="Load CSV", command=self.load_csv).pack(fill=tk.X, pady=5)

        # --- Sampling Frequency (AUTOMATIC) ---
        ttk.Label(self.sidebar, text="Sampling Freq (Hz) [Auto-Detected]:").pack(fill=tk.X)
        self.sfreq_var = tk.StringVar(value="N/A")
        sfreq_entry = ttk.Entry(self.sidebar, textvariable=self.sfreq_var, state='readonly')
        sfreq_entry.pack(fill=tk.X, pady=(0, 10))

        # --- Dataset and Trial Selection ---
        ttk.Label(self.sidebar, text="Dataset:").pack(fill=tk.X)
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(self.sidebar, textvariable=self.dataset_var, state="readonly")
        self.dataset_dropdown.pack(fill=tk.X, pady=(0, 5))
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dataset_select)

        ttk.Label(self.sidebar, text="Trial:").pack(fill=tk.X)
        self.trial_var = tk.StringVar()
        self.trial_dropdown = ttk.Combobox(self.sidebar, textvariable=self.trial_var, state="readonly")
        self.trial_dropdown.pack(fill=tk.X, pady=(0, 5))
        self.trial_dropdown.bind("<<ComboboxSelected>>", self.on_trial_select)
        
        ttk.Button(self.sidebar, text="Refresh Trials", command=self.update_trial_dropdown).pack(fill=tk.X, pady=5)

        # --- Channel Selection UI (Tabbed) ---
        self.channel_notebook = ttk.Notebook(self.sidebar)
        self.channel_notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        eeg_tab = ttk.Frame(self.channel_notebook)
        emg_tab = ttk.Frame(self.channel_notebook)
        self.channel_notebook.add(eeg_tab, text='EEG Channels')
        self.channel_notebook.add(emg_tab, text='EMG Channels')

        self.eeg_listbox = tk.Listbox(eeg_tab, selectmode=tk.MULTIPLE, exportselection=False)
        self.eeg_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.emg_listbox = tk.Listbox(emg_tab, selectmode=tk.MULTIPLE, exportselection=False)
        self.emg_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Bandpass Filter ---
        filter_frame = ttk.LabelFrame(self.sidebar, text="Bandpass Filter (Hz)")
        filter_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        
        ttk.Label(filter_frame, text="Low Cutoff:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lowcut_var = tk.StringVar(value="0.1")
        ttk.Entry(filter_frame, textvariable=self.lowcut_var, width=8).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(filter_frame, text="High Cutoff:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.highcut_var = tk.StringVar(value="30")
        ttk.Entry(filter_frame, textvariable=self.highcut_var, width=8).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(filter_frame, text="Apply Filter & Plot", command=self.apply_and_plot).grid(row=2, column=0, columnspan=2, pady=5)

        # --- Event Labeling ---
        event_frame = ttk.Frame(self.sidebar)
        event_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        ttk.Label(event_frame, text="Event Label:").pack(fill=tk.X, pady=(10, 0))
        self.event_label_var = tk.StringVar(value="Event1")
        ttk.Entry(event_frame, textvariable=self.event_label_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(event_frame, text="Labeled Events:").pack(fill=tk.X)
        self.events_listbox = tk.Listbox(event_frame, height=6)
        self.events_listbox.pack(fill=tk.X, expand=True, pady=(0, 10))

        ttk.Button(event_frame, text="Delete Selected Event", command=self.delete_event).pack(fill=tk.X, pady=5)
        ttk.Button(event_frame, text="Save Labeled Data", command=self.save_data).pack(fill=tk.X, pady=(10, 5))

    def _create_plot_widgets(self):
        """Creates the Matplotlib figure and canvas for plotting."""
        self.fig = Figure(figsize=(10, 8))
        self.ax_eeg = self.fig.add_subplot(211)
        self.ax_emg = self.fig.add_subplot(212, sharex=self.ax_eeg)
        self.fig.tight_layout(pad=4.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_panel)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.draw()
        
        self.span_selector = SpanSelector(
            self.ax_eeg, self.on_span_select, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='red'), interactive=True, drag_from_anywhere=True
        )

    def load_csv(self):
        """Loads a CSV file and populates UI elements, including default channel selections."""
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not filepath: return

        try:
            self.df = pd.read_csv(filepath)
            
            if not all(k in self.df for k in ['dataset', 'trial_number', 'time']):
                messagebox.showerror("Error", "CSV must contain 'dataset', 'trial_number', and 'time' columns.")
                return

            if 'event_label' not in self.df.columns: self.df['event_label'] = ''
            else: self.df['event_label'] = self.df['event_label'].fillna('')

            self.eeg_channels = sorted([col for col in self.df.columns if col.startswith('eeg')])
            self.emg_channels = sorted([col for col in self.df.columns if col.startswith('emg')])
            
            # --- Populate and set defaults for EEG channels ---
            self.eeg_listbox.delete(0, tk.END)
            default_eeg = {'9', '16', '18'}
            for ch in self.eeg_channels:
                self.eeg_listbox.insert(tk.END, ch)
                # Check if the number part of the channel name is in our defaults
                num_part = re.findall(r'\d+', ch)
                if num_part and num_part[0] in default_eeg:
                    self.eeg_listbox.selection_set(tk.END)

            # --- Populate and set defaults for EMG channels ---
            self.emg_listbox.delete(0, tk.END)
            default_emg = {'1', '2'}
            for ch in self.emg_channels:
                self.emg_listbox.insert(tk.END, ch)
                num_part = re.findall(r'\d+', ch)
                if num_part and num_part[0] in default_emg:
                    self.emg_listbox.selection_set(tk.END)

            datasets = sorted(self.df['dataset'].unique())
            self.dataset_dropdown['values'] = datasets
            if datasets:
                self.dataset_var.set(datasets[0])
                self.on_dataset_select()
                
        except Exception as e:
            messagebox.showerror("Error Loading File", f"An error occurred: {e}")

    def on_dataset_select(self, event=None):
        """Handles dataset selection and updates the trial dropdown."""
        self.update_trial_dropdown()

    def update_trial_dropdown(self):
        """Populates the trial dropdown based on the selected dataset."""
        if self.df is None: return
            
        selected_dataset = self.dataset_var.get()
        trials = sorted(self.df[self.df['dataset'] == selected_dataset]['trial_number'].unique())
        
        print(f"DEBUG: Found trials for dataset '{selected_dataset}': {trials}")

        if trials:
            self.trial_dropdown['values'] = trials
            self.trial_var.set(trials[0])
            self.on_trial_select()
        else:
            messagebox.showwarning("No Trials", f"No trials found for dataset '{selected_dataset}'.")
            self.trial_dropdown['values'] = []
            self.trial_var.set('')
            self.clear_plots()

    def on_trial_select(self, event=None):
        """Handles trial selection and triggers plotting."""
        if self.load_trial_data():
            self.apply_and_plot()

    def load_trial_data(self):
        """Loads data for the current trial and detects sampling rate. Returns True on success."""
        if self.df is None or not self.trial_var.get():
            self.current_trial_data = None
            return False

        try:
            selected_dataset = self.dataset_var.get()
            selected_trial = int(self.trial_var.get())
            
            self.current_trial_data = self.df[
                (self.df['dataset'] == selected_dataset) & (self.df['trial_number'] == selected_trial)
            ].copy()

            if not self.current_trial_data.empty and 'time' in self.current_trial_data.columns:
                time_data = self.current_trial_data['time'].values
                if len(time_data) > 1:
                    time_diff = np.mean(np.diff(time_data))
                    if time_diff > 1e-9: # Check for non-zero time difference
                        self.calculated_sfreq = 1000.0 / time_diff
                        self.sfreq_var.set(f"{self.calculated_sfreq:.2f}")
                    else:
                        self.calculated_sfreq = None; self.sfreq_var.set("Error: Time not increasing")
                else:
                    self.calculated_sfreq = None; self.sfreq_var.set("N/A: Not enough data")
            else:
                 self.calculated_sfreq = None; self.sfreq_var.set("N/A")

            self.update_events_listbox()
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Could not load trial data: {e}")
            self.current_trial_data = None
            return False

    def apply_and_plot(self):
        """Validates settings and plots data for the current trial."""
        if self.current_trial_data is None or self.current_trial_data.empty:
            self.clear_plots()
            return

        if self.calculated_sfreq is None:
            messagebox.showerror("Error", "Sampling frequency could not be determined. Cannot plot.")
            return
        sfreq = self.calculated_sfreq
            
        try:
            lowcut = float(self.lowcut_var.get())
            highcut = float(self.highcut_var.get())
            
            if not (lowcut > 0 and highcut > lowcut and highcut < sfreq / 2):
                messagebox.showerror("Filter Error", f"Invalid filter params. Ensure lowcut > 0, highcut > lowcut, and highcut < {sfreq / 2:.1f} Hz.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Cutoffs must be numeric."); return

        # Get selected EEG channels
        selected_eeg_ch_indices = self.eeg_listbox.curselection()
        selected_eeg_channels = [self.eeg_listbox.get(i) for i in selected_eeg_ch_indices]
        if not selected_eeg_channels: messagebox.showwarning("Warning", "No EEG channels selected.")
        
        # Get selected EMG channels
        selected_emg_ch_indices = self.emg_listbox.curselection()
        selected_emg_channels = [self.emg_listbox.get(i) for i in selected_emg_ch_indices]
        if not selected_emg_channels: messagebox.showwarning("Warning", "No EMG channels selected.")
        
        self.plot_data(selected_eeg_channels, selected_emg_channels, sfreq, lowcut, highcut)

    def plot_data(self, eeg_channels_to_plot, emg_channels_to_plot, sfreq, lowcut, highcut):
        """Plots selected EEG and EMG signals."""
        self.ax_eeg.clear(); self.ax_emg.clear()
        time_axis = self.current_trial_data['time']

        # --- Plot Filtered EEG ---
        for ch in eeg_channels_to_plot:
            raw_eeg = self.current_trial_data[ch].values
            filtered_eeg = apply_filter(raw_eeg, lowcut, highcut, sfreq)
            self.ax_eeg.plot(time_axis, filtered_eeg, label=ch, lw=1)
        self.ax_eeg.set_title(f"Filtered EEG: {self.dataset_var.get()}, Trial: {self.trial_var.get()}, Bandpass: {lowcut}-{highcut} Hz")
        self.ax_eeg.set_ylabel("Amplitude (filtered)"); self.ax_eeg.legend(loc='upper right', fontsize='small'); self.ax_eeg.grid(True)
        
        # --- Plot Selected EMG ---
        for ch in emg_channels_to_plot:
            self.ax_emg.plot(time_axis, self.current_trial_data[ch], label=ch, lw=1)
        self.ax_emg.set_title("Selected EMG Signals")
        self.ax_emg.set_xlabel("Time (ms)"); self.ax_emg.set_ylabel("Amplitude"); self.ax_emg.legend(loc='upper right', fontsize='small'); self.ax_emg.grid(True)
        
        self.fig.tight_layout(pad=4.0)
        self.redraw_spans()
        self.canvas.draw()

    def on_span_select(self, xmin, xmax):
        """Callback for when a span is selected on the EEG plot."""
        label = self.event_label_var.get().strip()
        if not label: messagebox.showwarning("Label Missing", "Please enter an event label."); return
        
        span_id = str(uuid.uuid4())
        span = self.ax_eeg.axvspan(xmin, xmax, facecolor='red', alpha=0.3)
        self.labeled_spans[span_id] = {'span_obj': span, 'label': label, 'xmin': xmin, 'xmax': xmax}
        
        listbox_text = f"{label}: {xmin:.2f} - {xmax:.2f} ({span_id})"
        self.events_listbox.insert(tk.END, listbox_text)
        
        self.update_df_with_label(xmin, xmax, label)
        print(f"Labeled span '{label}' from {xmin:.2f}s to {xmax:.2f}s.")

    def update_df_with_label(self, xmin, xmax, label):
        """Updates the event_label column in the main DataFrame."""
        if self.df is None: return
        
        mask = (self.df['dataset'] == self.dataset_var.get()) & \
               (self.df['trial_number'] == int(self.trial_var.get())) & \
               (self.df['time'] >= xmin) & (self.df['time'] <= xmax)
        self.df.loc[mask, 'event_label'] = label

    def delete_event(self):
        """Deletes the selected event from the listbox, plot, and DataFrame."""
        selected_indices = self.events_listbox.curselection()
        if not selected_indices: messagebox.showwarning("Warning", "No event selected to delete."); return

        listbox_text = self.events_listbox.get(selected_indices[0])
        span_id = listbox_text.split('(')[-1].strip(')')

        if span_id in self.labeled_spans:
            span_info = self.labeled_spans.pop(span_id)
            self.update_df_with_label(span_info['xmin'], span_info['xmax'], '')
            if span_info['span_obj'] in self.ax_eeg.patches: span_info['span_obj'].remove()
            self.events_listbox.delete(selected_indices[0])
            self.canvas.draw()
            print(f"Deleted event with ID: {span_id}")
        else:
            messagebox.showerror("Error", "Could not find the selected event to delete. Please refresh.")

    def update_events_listbox(self):
        """Clears and re-populates the events listbox from the DataFrame for the current trial."""
        self.events_listbox.delete(0, tk.END); self.labeled_spans.clear()
        if self.current_trial_data is None: return

        labeled_data = self.current_trial_data[self.current_trial_data['event_label'].notna() & (self.current_trial_data['event_label'] != '')]
        if labeled_data.empty: return

        for _, group in labeled_data.groupby((labeled_data['event_label'] != labeled_data['event_label'].shift()).cumsum()):
            start_time, end_time, event_label = group['time'].iloc[0], group['time'].iloc[-1], group['event_label'].iloc[0]
            span_id = str(uuid.uuid4())
            self.events_listbox.insert(tk.END, f"{event_label}: {start_time:.2f} - {end_time:.2f} ({span_id})")
            self.labeled_spans[span_id] = {'span_obj': None, 'label': event_label, 'xmin': start_time, 'xmax': end_time}
        
        self.redraw_spans()

    def redraw_spans(self):
        """Redraws all labeled spans on the EEG plot."""
        for patch in self.ax_eeg.patches: patch.remove()
        for span_id, info in self.labeled_spans.items():
            span = self.ax_eeg.axvspan(info['xmin'], info['xmax'], facecolor='red', alpha=0.3)
            info['span_obj'] = span
        self.canvas.draw()

    def save_data(self):
        """Saves the modified DataFrame to a new CSV file."""
        if self.df is None: messagebox.showerror("Error", "No data loaded to save."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        try:
            self.df.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Labeled data successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {e}")

    def clear_plots(self):
        """Clears the plot axes."""
        self.ax_eeg.clear(); self.ax_eeg.set_title("Filtered EEG"); self.ax_eeg.grid(True)
        self.ax_emg.clear(); self.ax_emg.set_title("Selected EMG Signals"); self.ax_emg.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGLabeler(root)
    root.mainloop()