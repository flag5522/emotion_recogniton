import csv
import os
import librosa
import numpy as np
import pandas as pd
import parselmouth
from librosa.util import peak_pick
from scipy.signal import find_peaks


# Function to extract F1 Formant using Parselmouth
def extract_f1(audio_file, frame_size, frame_shift):
    sound = parselmouth.Sound(audio_file)
    formants = sound.to_formant_burg()
    num_frames = formants.get_number_of_frames()
    f1_values = []

    for frame in range(num_frames):
        f1_value = formants.get_value_at_time(formant_number=1, time=frame * frame_shift)
        f1_values.append(f1_value)

    return f1_values


# Function to extract F2 Formant using Parselmouth
def extract_f2(audio_file, frame_size, frame_shift):
    sound = parselmouth.Sound(audio_file)
    formants = sound.to_formant_burg()
    num_frames = formants.get_number_of_frames()
    f2_values = []

    for frame in range(num_frames):
        f2_value = formants.get_value_at_time(formant_number=2, time=frame * frame_shift)
        f2_values.append(f2_value)

    return f2_values


# Function to extract Energy
def extract_energy(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    energy = librosa.feature.rms(y=y, frame_length=int(frame_size * sr), hop_length=int(frame_shift * sr))
    return energy[0]


# Function to extract MFCC
def extract_mfcc(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=int(frame_size * sr), hop_length=int(frame_shift * sr), n_mfcc=13)

    # Calculate the mean along the time axis (axis=1)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# Function to extract Spectral Magnitude
def extract_spectral_magnitude(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    spectral_magnitudes = []
    hop_s = int(frame_shift * sr)

    for i in range(0, len(y) - int(frame_size * sr), hop_s):
        frame = y[i:i + int(frame_size * sr)]
        spec = np.abs(librosa.stft(frame))
        magnitude = np.sum(spec)
        spectral_magnitudes.append(magnitude)

    return spectral_magnitudes

def extract_spectral_peaks(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    spectral_peaks = []

    hop_s = int(frame_shift * sr)

    for i in range(0, len(y) - int(frame_size * sr), hop_s):
        frame = y[i:i + int(frame_size * sr)]
        spec = np.abs(librosa.stft(frame))

        # Find peaks in the magnitude spectrum using scipy
        peaks, _ = find_peaks(spec.max(axis=0))

        # Append the peak frequencies to the list
        peak_frequencies = librosa.fft_frequencies(sr=sr)
        spectral_peaks.append(peak_frequencies[peaks])

    return spectral_peaks


# Function to extract Speaking Rate
def extract_speaking_rate(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    speaking_rate = len(y) / (frame_shift * sr)

    return speaking_rate


# Function to extract Pitch Peaks
def extract_pitch_peaks(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=int(frame_size * sr), hop_length=int(frame_shift * sr))

    # Find the pitch with the maximum magnitude in each frame
    pitch_peaks = np.argmax(magnitudes, axis=0)

    return pitch_peaks


# Function to extract Loudness
def extract_loudness(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    loudness = librosa.feature.rms(y=y, frame_length=int(frame_size * sr), hop_length=int(frame_shift * sr))

    return loudness[0]


# Function to extract Pitch Counter
def extract_pitch_counter(audio_file, frame_size, frame_shift):
    y, sr = librosa.load(audio_file, sr=None)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=int(frame_size * sr), hop_length=int(frame_shift * sr))

    # Count the number of non-zero pitches in each frame
    pitch_counter = np.sum(pitches > 0, axis=0)

    return pitch_counter


input_directory = "./anger/"

# Set the frame size and shift
frame_size = 0.2  # 20ms in seconds
frame_shift = 0.07  # 7ms in seconds


def mpm_f0_estimation(y, sr):
    # Compute the mean-shifted power spectrum
    n_fft = 512  # Use a smaller n_fft value
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Find the peak frequencies using mean-shift
    freqs, _, _ = librosa.reassigned_spectrogram(S_dB)

    # Flatten the freqs array to 1-D
    freqs = freqs.flatten()

    # Pick peaks in the mean-shifted power spectrum using peak_pick
    peaks = peak_pick(freqs, pre_max=20, post_max=20, pre_avg=20, post_avg=20, delta=0.5, wait=10)

    # Extract the frequencies corresponding to the peaks
    f0_mpm = freqs[peaks]

    return f0_mpm


def music_f0_estimation(y, sr):
    # Compute the autocorrelation function
    ac = librosa.autocorrelate(y)

    # Estimate the fundamental frequency using MUSIC algorithm
    f0_music = librosa.estimate_tuning(y=y, sr=sr)

    return f0_music


def save_to_csv(filename, data, header):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Header
        writer.writerows(data)


def process_directory(directory_path):
    output_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # MPM F0 estimation
            f0_mpm = mpm_f0_estimation(y, sr)

            # MUSIC F0 estimation
            f0_music = music_f0_estimation(y, sr)

            # Append the data to the output list
            output_data.append([np.mean(f0_mpm), f0_music])

    # Save the data to a CSV file
    csv_filename = 'apeaks.csv'
    save_to_csv(csv_filename, output_data, header=['MPM', 'MUSIC'])


process_directory(input_directory)

# Initialize lists to store feature values
all_f1_values = []
all_f2_values = []
all_energy_values = []
all_mfcc_values = []
all_spectral_peaks = []
all_spectral_magnitudes = []
all_speaking_rates = []
all_pitch_peaks = []
all_loudness_values = []
all_pitch_counters = []

# List all WAV files in the directory
audio_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if
               filename.endswith(".wav")]

# Loop through each audio file and extract the features
for audio_file in audio_files:
    print("Processing:", audio_file)

    # Extract the features
    f1_values = extract_f1(audio_file, frame_size, frame_shift)
    f2_values = extract_f2(audio_file, frame_size, frame_shift)
    energy_values = extract_energy(audio_file, frame_size, frame_shift)
    mfcc_values = extract_mfcc(audio_file, frame_size, frame_shift)
    spectral_magnitudes = extract_spectral_magnitude(audio_file, frame_size, frame_shift)
    spectral_peaks = extract_spectral_peaks(audio_file, frame_size, frame_shift)
    speaking_rate = extract_speaking_rate(audio_file, frame_size, frame_shift)
    pitch_peaks = extract_pitch_peaks(audio_file, frame_size, frame_shift)
    loudness_values = extract_loudness(audio_file, frame_size, frame_shift)
    pitch_counter = extract_pitch_counter(audio_file, frame_size, frame_shift)

    # Print or use the extracted features as needed
    print("F1 values:", f1_values)
    print("F2 values:", f2_values)
    print("Energy values:", energy_values)
    print("MFCC values:", mfcc_values)
    print("Spectral Magnitudes:", spectral_magnitudes)
    print("Spectral Peaks:", spectral_peaks)
    print("Speaking Rate:", speaking_rate)
    print("Pitch Peaks:", pitch_peaks)
    print("Loudness Values:", loudness_values)
    print("Pitch Counter:", pitch_counter)

    all_f1_values.extend(f1_values)
    all_f2_values.extend(f2_values)
    all_energy_values.extend(energy_values)
    all_mfcc_values.append(mfcc_values)
    all_spectral_magnitudes.extend(spectral_magnitudes)
    all_spectral_peaks.extend(spectral_peaks)
    all_speaking_rates.append(speaking_rate)
    all_pitch_peaks.extend(pitch_peaks)
    all_loudness_values.extend(loudness_values)
    all_pitch_counters.extend(pitch_counter)

# Store the extracted feature values in a CSV file
csv_filename = "afeatures.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["F1 Values",
                     "F2 Values",
                     "Energy Values",
                     "MFCC Values",
                     "Spectral Magnitudes",
                     "Spectral Peaks",
                     "Speaking Rate",
                     "Pitch Peaks",
                     "Loudness Values",
                     "Pitch Counter"])
    writer.writerows(
        zip(all_f1_values,
            all_f2_values,
            all_energy_values,
            all_mfcc_values,
            all_spectral_magnitudes,
            all_spectral_peaks,
            all_speaking_rates,
            all_pitch_peaks,
            all_loudness_values,
            all_pitch_counters))

print("Feature values saved to 'features.csv'.")

# Load CSV files into DataFrames
peaks_df = pd.read_csv("apeaks.csv")
features_df = pd.read_csv("afeatures.csv")

# Specify the order of columns in the combined DataFrame
column_order = ['MPM', 'MUSIC', 'F1 Values', 'F2 Values', 'Energy Values', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
                'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13',
                'Spectral Magnitudes', 'Spectral Peaks', 'Speaking Rate', 'Pitch Peaks', 'Loudness Values',
                'Pitch Counter']

# Create a new DataFrame with the specified column order
combined_df = pd.concat([peaks_df['MPM'], peaks_df['MUSIC'], features_df], axis=1)

# Split the "MFCC Values" column into 13 separate columns
mfcc_columns = pd.DataFrame(combined_df['MFCC Values'].apply(lambda x: pd.Series(map(float, x.strip('[]').split()))))

# Rename the MFCC columns
mfcc_columns.columns = [f'MFCC {i + 1}' for i in range(13)]

# Concatenate the MFCC columns with the combined DataFrame
combined_df = pd.concat([combined_df, mfcc_columns], axis=1)

# Drop the original "MFCC Values" column
combined_df.drop('MFCC Values', axis=1, inplace=True)

# Extract the largest number from the "Spectral Peaks" column
combined_df['Spectral Peaks'] = combined_df['Spectral Peaks'].apply(lambda x: max(map(float, x.strip('[]').split())))

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("a_combined_data.csv", index=False)
