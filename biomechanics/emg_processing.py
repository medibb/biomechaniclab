"""
OpenBCI EMG Signal Processing Pipeline

EMG signal processing utilities for OpenBCI Cyton/Ganglion boards.
Based on rehabilitation biomechanics practical guide.

Supported boards:
- Cyton: 8ch (expandable to 16ch), 250Hz sampling
- Ganglion: 4ch, 200Hz sampling
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, List


def process_emg(raw_emg: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    OpenBCI EMG signal processing pipeline.

    Parameters
    ----------
    raw_emg : np.ndarray
        Raw EMG signal
    fs : int
        Sampling frequency (Cyton: 250Hz, Ganglion: 200Hz)

    Returns
    -------
    np.ndarray
        Linear envelope of the EMG signal
    """
    # 1. Bandpass filter (20-120Hz for OpenBCI, limited by Nyquist)
    nyq = fs / 2
    low = 20 / nyq
    high = min(120, nyq - 1) / nyq  # Limit to below Nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, raw_emg)

    # 2. Full-wave rectification
    rectified = np.abs(filtered)

    # 3. Lowpass filter for linear envelope (6Hz cutoff)
    low_env = 6 / nyq
    b_env, a_env = signal.butter(4, low_env, btype='low')
    envelope = signal.filtfilt(b_env, a_env, rectified)

    return envelope


def normalize_emg(envelope: np.ndarray, mvc_value: float) -> np.ndarray:
    """
    Normalize EMG envelope to %MVC.

    Parameters
    ----------
    envelope : np.ndarray
        EMG envelope signal
    mvc_value : float
        Maximum voluntary contraction value

    Returns
    -------
    np.ndarray
        Normalized EMG (%MVC)
    """
    return (envelope / mvc_value) * 100


def calculate_mvc(trials: List[np.ndarray], fs: int = 250,
                  window_sec: float = 0.5) -> float:
    """
    Calculate MVC from multiple trial recordings.

    Uses the peak of the moving average across all trials.

    Parameters
    ----------
    trials : List[np.ndarray]
        List of MVC trial recordings
    fs : int
        Sampling frequency
    window_sec : float
        Moving average window in seconds

    Returns
    -------
    float
        MVC value
    """
    window_samples = int(window_sec * fs)
    max_values = []

    for trial in trials:
        envelope = process_emg(trial, fs)
        # Moving average
        kernel = np.ones(window_samples) / window_samples
        smoothed = np.convolve(envelope, kernel, mode='valid')
        max_values.append(np.max(smoothed))

    return np.max(max_values)


def calculate_activation_timing(
    envelope: np.ndarray,
    threshold: float = 0.1,
    fs: int = 250
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect muscle activation onset/offset timing.

    Parameters
    ----------
    envelope : np.ndarray
        EMG envelope signal
    threshold : float
        Activation threshold (ratio of maximum)
    fs : int
        Sampling frequency

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        onset indices, offset indices
    """
    thresh_value = threshold * np.max(envelope)
    active = envelope > thresh_value

    # Find onset/offset points
    diff = np.diff(active.astype(int))
    onset = np.where(diff == 1)[0]
    offset = np.where(diff == -1)[0]

    return onset, offset


def calculate_rms(emg_signal: np.ndarray, window_samples: int = 50) -> np.ndarray:
    """
    Calculate Root Mean Square of EMG signal.

    Parameters
    ----------
    emg_signal : np.ndarray
        Raw or filtered EMG signal
    window_samples : int
        RMS window size in samples

    Returns
    -------
    np.ndarray
        RMS envelope
    """
    squared = emg_signal ** 2
    kernel = np.ones(window_samples) / window_samples
    mean_squared = np.convolve(squared, kernel, mode='same')
    return np.sqrt(mean_squared)


def calculate_mean_frequency(emg_signal: np.ndarray, fs: int = 250) -> float:
    """
    Calculate mean frequency of EMG signal (fatigue indicator).

    Parameters
    ----------
    emg_signal : np.ndarray
        Raw EMG signal segment
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Mean frequency in Hz
    """
    freqs, psd = signal.welch(emg_signal, fs, nperseg=min(256, len(emg_signal)))
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    return mean_freq


def calculate_median_frequency(emg_signal: np.ndarray, fs: int = 250) -> float:
    """
    Calculate median frequency of EMG signal (fatigue indicator).

    Parameters
    ----------
    emg_signal : np.ndarray
        Raw EMG signal segment
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Median frequency in Hz
    """
    freqs, psd = signal.welch(emg_signal, fs, nperseg=min(256, len(emg_signal)))
    cumsum = np.cumsum(psd)
    median_idx = np.where(cumsum >= cumsum[-1] / 2)[0][0]
    return freqs[median_idx]


def calculate_cocontraction_index(
    agonist: np.ndarray,
    antagonist: np.ndarray
) -> np.ndarray:
    """
    Calculate co-contraction index between agonist and antagonist muscles.

    Parameters
    ----------
    agonist : np.ndarray
        Agonist muscle EMG envelope
    antagonist : np.ndarray
        Antagonist muscle EMG envelope

    Returns
    -------
    np.ndarray
        Co-contraction index (0-1)
    """
    min_activity = np.minimum(agonist, antagonist)
    max_activity = np.maximum(agonist, antagonist)

    # Avoid division by zero
    cci = (2 * min_activity) / (agonist + antagonist + 1e-10)

    return cci


def remove_ecg_artifact(
    emg_signal: np.ndarray,
    fs: int = 250,
    template_method: bool = True
) -> np.ndarray:
    """
    Remove ECG artifacts from EMG signal.

    Parameters
    ----------
    emg_signal : np.ndarray
        Raw EMG signal with ECG contamination
    fs : int
        Sampling frequency
    template_method : bool
        Use template subtraction method

    Returns
    -------
    np.ndarray
        EMG signal with reduced ECG artifact
    """
    if template_method:
        # Simple highpass filter to reduce ECG
        nyq = fs / 2
        high = 30 / nyq
        b, a = signal.butter(2, high, btype='high')
        cleaned = signal.filtfilt(b, a, emg_signal)
        return cleaned
    else:
        # Notch filter for power line interference
        notch_freq = 60.0  # or 50.0 for non-US
        quality = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality, fs)
        return signal.filtfilt(b_notch, a_notch, emg_signal)


class EMGProcessor:
    """
    EMG processing class for multi-channel data.

    Attributes
    ----------
    fs : int
        Sampling frequency
    mvc_values : Dict[str, float]
        MVC values for each channel
    channel_names : List[str]
        Names of EMG channels
    """

    # Standard muscle configurations
    SHOULDER_MUSCLES = [
        'upper_trapezius', 'middle_trapezius', 'anterior_deltoid',
        'middle_deltoid', 'supraspinatus', 'infraspinatus',
        'pectoralis_major', 'serratus_anterior'
    ]

    LUMBAR_MUSCLES = [
        'erector_spinae_L3_L', 'erector_spinae_L3_R',
        'multifidus_L5_L', 'multifidus_L5_R',
        'external_oblique_L', 'external_oblique_R',
        'rectus_abdominis', 'gluteus_maximus'
    ]

    KNEE_MUSCLES = [
        'VMO', 'VL', 'RF', 'medial_hamstring',
        'lateral_hamstring', 'gastrocnemius_medial',
        'tibialis_anterior', 'gluteus_maximus'
    ]

    def __init__(self, fs: int = 250, channel_names: Optional[List[str]] = None):
        """
        Initialize EMG processor.

        Parameters
        ----------
        fs : int
            Sampling frequency
        channel_names : List[str], optional
            Names for each channel
        """
        self.fs = fs
        self.mvc_values: Dict[str, float] = {}
        self.channel_names = channel_names or []

    def set_mvc(self, channel: str, mvc_value: float):
        """Set MVC value for a channel."""
        self.mvc_values[channel] = mvc_value

    def process_multichannel(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process multi-channel EMG data.

        Parameters
        ----------
        data : np.ndarray
            Multi-channel EMG data (channels x samples)
        normalize : bool
            Whether to normalize to %MVC

        Returns
        -------
        Dict[str, np.ndarray]
            Processed EMG for each channel
        """
        results = {}

        for i, channel_data in enumerate(data):
            channel_name = (self.channel_names[i]
                          if i < len(self.channel_names)
                          else f'channel_{i}')

            envelope = process_emg(channel_data, self.fs)

            if normalize and channel_name in self.mvc_values:
                envelope = normalize_emg(envelope, self.mvc_values[channel_name])

            results[channel_name] = envelope

        return results

    def analyze_activation_pattern(
        self,
        processed_data: Dict[str, np.ndarray],
        threshold: float = 0.1
    ) -> Dict[str, Dict]:
        """
        Analyze activation patterns for all channels.

        Parameters
        ----------
        processed_data : Dict[str, np.ndarray]
            Processed EMG data
        threshold : float
            Activation threshold

        Returns
        -------
        Dict[str, Dict]
            Activation timing and statistics for each channel
        """
        results = {}

        for channel, envelope in processed_data.items():
            onset, offset = calculate_activation_timing(envelope, threshold, self.fs)

            results[channel] = {
                'onset_samples': onset,
                'offset_samples': offset,
                'onset_times': onset / self.fs,
                'offset_times': offset / self.fs,
                'peak_amplitude': np.max(envelope),
                'mean_amplitude': np.mean(envelope),
                'duration': len(envelope) / self.fs
            }

        return results
