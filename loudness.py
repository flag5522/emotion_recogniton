import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data CSV files for happiness, sadness, and anger
happiness_data = pd.read_csv('h_combined_data.csv')
sadness_data = pd.read_csv('s_combined_data.csv')
anger_data = pd.read_csv('a_combined_data.csv')

# Extract the loudness columns
happiness_loudness = happiness_data['Loudness Values']
sadness_loudness = sadness_data['Loudness Values']
anger_loudness = anger_data['Loudness Values']

# Calculate the average loudness values
average_happiness_loudness = np.mean(happiness_loudness)
average_sadness_loudness = np.mean(sadness_loudness)
average_anger_loudness = np.mean(anger_loudness)

# Plot a graph of the average loudness values
categories = ['Happiness', 'Sadness', 'Anger']
average_loudness_values = [average_happiness_loudness, average_sadness_loudness, average_anger_loudness]

plt.bar(categories, average_loudness_values, color=['orange', 'purple', 'yellow'])
plt.title('Average Loudness Comparison')
plt.xlabel('Emotion')
plt.ylabel('Average Loudness')
plt.show()
