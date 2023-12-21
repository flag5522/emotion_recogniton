import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data CSV files for happiness, sadness, and anger
happiness_data = pd.read_csv('h_combined_data.csv')
sadness_data = pd.read_csv('s_combined_data.csv')
anger_data = pd.read_csv('a_combined_data.csv')

# Extract the MPM columns
happiness_mpm = happiness_data['MPM']
sadness_mpm = sadness_data['MPM']
anger_mpm = anger_data['MPM']

# Calculate the average MPM values
average_happiness_mpm = np.mean(happiness_mpm)
average_sadness_mpm = np.mean(sadness_mpm)
average_anger_mpm = np.mean(anger_mpm)

# Plot a graph of the average MPM values
categories = ['Happiness', 'Sadness', 'Anger']
average_mpm_values = [average_happiness_mpm, average_sadness_mpm, average_anger_mpm]

plt.bar(categories, average_mpm_values, color=['blue', 'green', 'red'])
plt.title('Average F0 Comparison')
plt.xlabel('Emotion')
plt.ylabel('Average F0')
plt.show()
