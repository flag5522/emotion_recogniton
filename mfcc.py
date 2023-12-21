import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data CSV files for happiness, sadness, and anger
happiness_data = pd.read_csv('h_combined_data.csv')
sadness_data = pd.read_csv('s_combined_data.csv')
anger_data = pd.read_csv('a_combined_data.csv')

# Extract the MFCC columns
happiness_mfcc = happiness_data.filter(regex='MFCC.*')
sadness_mfcc = sadness_data.filter(regex='MFCC.*')
anger_mfcc = anger_data.filter(regex='MFCC.*')

# Calculate the average MFCC values
average_happiness_mfcc = happiness_mfcc.mean(axis=0)
average_sadness_mfcc = sadness_mfcc.mean(axis=0)
average_anger_mfcc = anger_mfcc.mean(axis=0)

# Plot a graph of the average MFCC values
plt.figure(figsize=(10, 6))
plt.plot(average_happiness_mfcc, label='Happiness', marker='o', linestyle='-', color='blue')
plt.plot(average_sadness_mfcc, label='Sadness', marker='o', linestyle='-', color='green')
plt.plot(average_anger_mfcc, label='Anger', marker='o', linestyle='-', color='red')

plt.title('Average MFCC Comparison')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Average MFCC Value')
plt.legend()
plt.grid(True)
plt.show()
