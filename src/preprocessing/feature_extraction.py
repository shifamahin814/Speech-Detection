import numpy as np

def extract_features(signal, sample_rate):
    """
    Extract features from audio signal.

    Parameters:
    signal (numpy.ndarray): Audio signal
    sample_rate (int): Sample rate of the audio signal

    Returns:
    dict: Extracted features
    """
    # Example features
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
    }
    return features
