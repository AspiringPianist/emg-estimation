import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Your CSV result files
file_2_input = r"C:\Users\rajes\Desktop\IIITB\emg-emg\2_1_results_from_text.csv"
file_3_input = r"C:\Users\rajes\Desktop\IIITB\emg-emg\3_1_results_from_text.csv"

channels = [f'emg{i}' for i in range(1, 9)]
coords = np.array([
    [0.0, 1.0],    # emg1 (top)
    [-0.7, 0.7],   # emg2 (top left)
    [-1.0, 0.0],   # emg3 (left)
    [-0.7, -0.7],  # emg4 (bottom left)
    [0.0, -1.0],   # emg5 (bottom)
    [0.7, -0.7],   # emg6 (bottom right)
    [1.0, 0.0],    # emg7 (right)
    [0.7, 0.7]     # emg8 (top right)
])

def load_and_select_top3(filepath):
    df = pd.read_csv(filepath)
    top3 = df.sort_values('correlation', ascending=False).head(3)
    return top3

def plot_emg_importance_heatmap(inputs, target, channels, coords, title, save_path=None):
    highlight = np.full(8, 0.3)
    for i, ch in enumerate(channels):
        if ch in inputs:
            highlight[i] = 1.0
        elif ch == target:
            highlight[i] = 0.7
    grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
    heatmap = griddata(coords, highlight, (grid_x, grid_y), method='cubic')
    plt.figure(figsize=(8, 8), dpi=200)
    plt.contourf(grid_x, grid_y, heatmap, levels=100, cmap='RdBu_r')
    plt.scatter(coords[:,0], coords[:,1], c='black', s=60, edgecolor='white', linewidth=1.5, zorder=3)
    for i, label in enumerate(channels):
        plt.text(coords[i,0], coords[i,1], label, fontsize=10, ha='center', va='center',
                 color='white', weight='bold',
                 bbox=dict(facecolor='black', edgecolor='none', pad=0.3), zorder=4)
    plt.title(title, fontsize=14, weight='bold')
    plt.axis('equal')
    plt.axis('off')
    plt.colorbar(label='Role: Inputs (1.0), Target (0.7), Other (0.3)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Load and print top 3 results
print("Top 3 for 2 input channels:")
top3_2 = load_and_select_top3(file_2_input)
print(top3_2[['input_channels', 'target_channel', 'r_squared', 'mse', 'correlation']])

print("\nTop 3 for 3 input channels:")
top3_3 = load_and_select_top3(file_3_input)
print(top3_3[['input_channels', 'target_channel', 'r_squared', 'mse', 'correlation']])

# Visualize and save heatmap for the top combo in each case
inputs_2 = [ch.strip() for ch in top3_2.iloc[0]['input_channels'].replace(' ', '').split(',')]
target_2 = top3_2.iloc[0]['target_channel']
plot_emg_importance_heatmap(inputs_2, target_2, channels, coords,
    f'Top 2-Input Combo: {inputs_2} → {target_2}', save_path="Top_2_input_combo.png")

inputs_3 = [ch.strip() for ch in top3_3.iloc[0]['input_channels'].replace(' ', '').split(',')]
target_3 = top3_3.iloc[0]['target_channel']
plot_emg_importance_heatmap(inputs_3, target_3, channels, coords,
    f'Top 3-Input Combo: {inputs_3} → {target_3}', save_path="Top_3_input_combo.png")
