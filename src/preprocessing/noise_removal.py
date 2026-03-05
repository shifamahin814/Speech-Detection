import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter

# Butterworth filter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply filter

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Noise removal function

def remove_noise(filepath, output_filepath, lowcut=300.0, highcut=3400.0):
    fs, data = wav.read(filepath)
    # Apply bandpass filter to remove noise
    filtered_data = bandpass_filter(data, lowcut, highcut, fs)
    wav.write(output_filepath, fs, filtered_data.astype(np.int16))

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python noise_removal.py <input_file.wav> <output_file.wav>')
        sys.exit(1)
    remove_noise(sys.argv[1], sys.argv[2])