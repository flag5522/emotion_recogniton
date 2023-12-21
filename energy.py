import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data CSV files for happiness, sadness, and anger
happiness_data = pd.read_csv('h_combined_data.csv')
sadness_data = pd.read_csv('s_combined_data.csv')
anger_data = pd.read_csv('a_combined_data.csv')

# Extract the energy columns
happiness_energy = happiness_data['Energy Values']
sadness_energy = sadness_data['Energy Values']
anger_energy = anger_data['Energy Values']

# Calculate the average energy values
average_happiness_energy = np.mean(happiness_energy)
average_sadness_energy = np.mean(sadness_energy)
average_anger_energy = np.mean(anger_energy)

# Plot a graph of the average energy values
categories = ['Happiness', 'Sadness', 'Anger']
average_energies = [average_happiness_energy, average_sadness_energy, average_anger_energy]

plt.bar(categories, average_energies, color=['blue', 'green', 'red'])
plt.title('Average Energy Comparison')
plt.xlabel('Emotion')
plt.ylabel('Average Energy')
plt.show()
