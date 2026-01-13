"""
Multi-Modal Data Recording and Synchronization Module

Unified data collection system for:
- Movella DOT IMU sensors (via LSL or direct)
- OpenBCI EMG (via LSL or Brainflow)
- Kinvent force plates (via timestamp-based sync)

Based on rehabilitation biomechanics practical guide.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class RecordingSession:
    """Container for a recording session."""
    session_id: str
    start_time: datetime
    duration_sec: float
    imu_data: Dict[str, np.ndarray] = field(default_factory=dict)
    emg_data: Dict[str, np.ndarray] = field(default_factory=dict)
    force_data: Dict[str, np.ndarray] = field(default_factory=dict)
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict = field(default_factory=dict)


class MultiModalRecorder:
    """
    Multi-modal data recorder using Lab Streaming Layer (LSL).

    Synchronizes data from:
    - Movella DOT IMU sensors
    - OpenBCI EMG/EEG
    - Kinvent force plates (timestamp-based)

    Attributes
    ----------
    streams : Dict
        Connected LSL streams
    data : Dict
        Recorded data buffers
    fs_target : int
        Target sampling frequency for resampling
    """

    def __init__(self, fs_target: int = 100):
        """
        Initialize multi-modal recorder.

        Parameters
        ----------
        fs_target : int
            Target sampling frequency for synchronized output
        """
        self.streams = {}
        self.data = {}
        self.fs_target = fs_target
        self._recording = False

    def connect_streams(self) -> Dict[str, bool]:
        """
        Connect to available LSL streams.

        Returns
        -------
        Dict[str, bool]
            Connection status for each stream type
        """
        try:
            from pylsl import StreamInlet, resolve_stream
        except ImportError:
            print("pylsl not installed. Install with: pip install pylsl")
            return {'imu': False, 'emg': False}

        status = {'imu': False, 'emg': False}

        # Search for IMU stream
        print("Searching for IMU stream...")
        try:
            imu_streams = resolve_stream('type', 'IMU', timeout=5.0)
            if imu_streams:
                self.streams['imu'] = StreamInlet(imu_streams[0])
                status['imu'] = True
                print(f"IMU stream connected: {imu_streams[0].name()}")
        except Exception as e:
            print(f"IMU stream not found: {e}")

        # Search for EMG stream
        print("Searching for EMG stream...")
        try:
            emg_streams = resolve_stream('type', 'EMG', timeout=5.0)
            if emg_streams:
                self.streams['emg'] = StreamInlet(emg_streams[0])
                status['emg'] = True
                print(f"EMG stream connected: {emg_streams[0].name()}")
        except Exception as e:
            print(f"EMG stream not found: {e}")

        return status

    def record(self, duration_sec: float) -> RecordingSession:
        """
        Record synchronized data for specified duration.

        Parameters
        ----------
        duration_sec : float
            Recording duration in seconds

        Returns
        -------
        RecordingSession
            Recorded data session
        """
        start_time = datetime.now()
        session_id = start_time.strftime("%Y%m%d_%H%M%S")

        self.data = {
            'imu': [],
            'emg': [],
            'timestamps': []
        }

        self._recording = True
        elapsed = 0

        while elapsed < duration_sec and self._recording:
            # Collect IMU data
            if 'imu' in self.streams:
                sample, timestamp = self.streams['imu'].pull_sample(timeout=0.0)
                if sample:
                    self.data['imu'].append(sample)
                    self.data['timestamps'].append(timestamp)

            # Collect EMG data
            if 'emg' in self.streams:
                sample, timestamp = self.streams['emg'].pull_sample(timeout=0.0)
                if sample:
                    self.data['emg'].append(sample)

            elapsed = (datetime.now() - start_time).total_seconds()

        session = RecordingSession(
            session_id=session_id,
            start_time=start_time,
            duration_sec=elapsed,
            imu_data={'raw': np.array(self.data['imu'])},
            emg_data={'raw': np.array(self.data['emg'])},
            timestamps=np.array(self.data['timestamps']),
            metadata={'fs_target': self.fs_target}
        )

        return session

    def stop_recording(self):
        """Stop ongoing recording."""
        self._recording = False

    def save_session(self, session: RecordingSession, filepath: str):
        """
        Save recording session to file.

        Parameters
        ----------
        session : RecordingSession
            Session to save
        filepath : str
            Output file path (without extension)
        """
        # Save numpy data
        np.savez(
            f"{filepath}.npz",
            imu=session.imu_data.get('raw', np.array([])),
            emg=session.emg_data.get('raw', np.array([])),
            timestamps=session.timestamps
        )

        # Save metadata
        metadata = {
            'session_id': session.session_id,
            'start_time': session.start_time.isoformat(),
            'duration_sec': session.duration_sec,
            'metadata': session.metadata
        }
        with open(f"{filepath}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Session saved to {filepath}")


def synchronize_offline(
    imu_file: str,
    emg_file: str,
    force_file: Optional[str] = None,
    sync_method: str = 'movement_onset'
) -> Dict[str, np.ndarray]:
    """
    Offline synchronization when data is collected separately.

    Uses movement onset detection to align different data streams.

    Parameters
    ----------
    imu_file : str
        Path to IMU data file
    emg_file : str
        Path to EMG data file
    force_file : str, optional
        Path to force plate data file
    sync_method : str
        Synchronization method: 'movement_onset', 'peak', 'manual'

    Returns
    -------
    Dict[str, np.ndarray]
        Synchronized data arrays
    """
    # Load data files
    imu_data = np.load(imu_file, allow_pickle=True)
    emg_data = np.load(emg_file, allow_pickle=True)

    force_data = None
    if force_file:
        force_data = np.load(force_file, allow_pickle=True)

    # Detect synchronization events
    if sync_method == 'movement_onset':
        # Detect sudden acceleration change in IMU
        imu_accel = imu_data.get('acceleration', imu_data.get('raw', np.zeros(100)))
        if len(imu_accel.shape) > 1:
            imu_accel = np.linalg.norm(imu_accel, axis=1)
        imu_sync = detect_movement_onset(imu_accel)

        # Detect EMG burst onset
        emg_raw = emg_data.get('raw', np.zeros(100))
        if len(emg_raw.shape) > 1:
            emg_envelope = np.mean(np.abs(emg_raw), axis=1)
        else:
            emg_envelope = np.abs(emg_raw)
        emg_sync = detect_emg_onset(emg_envelope)

        # Calculate time offsets
        offset_emg = imu_sync - emg_sync

        # Apply offsets
        emg_aligned = shift_data(emg_raw, offset_emg)

        if force_data is not None:
            force_raw = force_data.get('raw', np.zeros(100))
            force_sync = detect_force_onset(force_raw)
            offset_force = imu_sync - force_sync
            force_aligned = shift_data(force_raw, offset_force)
        else:
            force_aligned = None

    else:
        # No alignment, assume already synchronized
        emg_aligned = emg_data.get('raw', np.array([]))
        force_aligned = force_data.get('raw') if force_data else None

    # Resample to common rate
    target_fs = 100  # Hz
    imu_resampled = resample_to_rate(
        imu_data.get('raw', np.array([])), target_fs
    )
    emg_resampled = resample_to_rate(emg_aligned, target_fs)

    result = {
        'imu': imu_resampled,
        'emg': emg_resampled,
        'fs': target_fs
    }

    if force_aligned is not None:
        result['force'] = resample_to_rate(force_aligned, target_fs)

    return result


def detect_movement_onset(
    acceleration: np.ndarray,
    threshold_factor: float = 3.0
) -> int:
    """
    Detect movement onset from acceleration data.

    Parameters
    ----------
    acceleration : np.ndarray
        Acceleration signal (magnitude)
    threshold_factor : float
        Threshold as multiple of baseline std

    Returns
    -------
    int
        Sample index of movement onset
    """
    # Use first 10% as baseline
    baseline_end = len(acceleration) // 10
    baseline = acceleration[:baseline_end]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    threshold = baseline_mean + threshold_factor * baseline_std

    # Find first crossing
    above_threshold = acceleration > threshold
    onset_indices = np.where(above_threshold)[0]

    if len(onset_indices) > 0:
        return onset_indices[0]
    return 0


def detect_emg_onset(
    emg_envelope: np.ndarray,
    threshold_factor: float = 3.0
) -> int:
    """
    Detect EMG burst onset.

    Parameters
    ----------
    emg_envelope : np.ndarray
        EMG envelope signal
    threshold_factor : float
        Threshold as multiple of baseline std

    Returns
    -------
    int
        Sample index of EMG onset
    """
    baseline_end = len(emg_envelope) // 10
    baseline = emg_envelope[:baseline_end]
    threshold = np.mean(baseline) + threshold_factor * np.std(baseline)

    above_threshold = emg_envelope > threshold
    onset_indices = np.where(above_threshold)[0]

    if len(onset_indices) > 0:
        return onset_indices[0]
    return 0


def detect_force_onset(
    force: np.ndarray,
    threshold: float = 10.0
) -> int:
    """
    Detect force plate contact.

    Parameters
    ----------
    force : np.ndarray
        Vertical force signal
    threshold : float
        Force threshold in N

    Returns
    -------
    int
        Sample index of force onset
    """
    if len(force.shape) > 1:
        force = np.sum(force, axis=1)

    above_threshold = force > threshold
    onset_indices = np.where(above_threshold)[0]

    if len(onset_indices) > 0:
        return onset_indices[0]
    return 0


def shift_data(data: np.ndarray, offset: int) -> np.ndarray:
    """
    Shift data by offset samples.

    Parameters
    ----------
    data : np.ndarray
        Data to shift
    offset : int
        Number of samples to shift (positive = delay)

    Returns
    -------
    np.ndarray
        Shifted data
    """
    if offset == 0:
        return data

    if len(data.shape) == 1:
        if offset > 0:
            return np.concatenate([np.zeros(offset), data[:-offset]])
        else:
            return np.concatenate([data[-offset:], np.zeros(-offset)])
    else:
        # Multi-dimensional
        if offset > 0:
            padding = np.zeros((offset, data.shape[1]))
            return np.vstack([padding, data[:-offset]])
        else:
            padding = np.zeros((-offset, data.shape[1]))
            return np.vstack([data[-offset:], padding])


def resample_to_rate(
    data: np.ndarray,
    target_fs: int,
    original_fs: Optional[int] = None
) -> np.ndarray:
    """
    Resample data to target sampling rate.

    Parameters
    ----------
    data : np.ndarray
        Data to resample
    target_fs : int
        Target sampling frequency
    original_fs : int, optional
        Original sampling frequency (estimated if not provided)

    Returns
    -------
    np.ndarray
        Resampled data
    """
    if len(data) == 0:
        return data

    from scipy import signal

    if original_fs is None:
        # Estimate original fs (assume 1 second of data)
        original_fs = len(data)

    # Calculate resampling ratio
    num_samples = int(len(data) * target_fs / original_fs)

    if len(data.shape) == 1:
        return signal.resample(data, num_samples)
    else:
        # Resample each channel
        resampled = np.zeros((num_samples, data.shape[1]))
        for i in range(data.shape[1]):
            resampled[:, i] = signal.resample(data[:, i], num_samples)
        return resampled


class DataQualityChecker:
    """
    Data quality assessment utilities.
    """

    @staticmethod
    def check_imu_quality(data: np.ndarray, fs: int = 120) -> Dict[str, any]:
        """
        Check IMU data quality.

        Parameters
        ----------
        data : np.ndarray
            IMU data
        fs : int
            Sampling frequency

        Returns
        -------
        Dict[str, any]
            Quality metrics
        """
        results = {
            'duration_sec': len(data) / fs,
            'missing_samples': np.sum(np.isnan(data)),
            'dropout_percent': np.sum(np.isnan(data)) / data.size * 100
        }

        # Check for drift (compare start vs end baseline)
        if len(data) > fs * 10:  # At least 10 seconds
            start_baseline = np.mean(data[:fs])
            end_baseline = np.mean(data[-fs:])
            results['drift'] = end_baseline - start_baseline
            results['drift_significant'] = abs(results['drift']) > 5  # degrees

        return results

    @staticmethod
    def check_emg_quality(data: np.ndarray, fs: int = 250) -> Dict[str, any]:
        """
        Check EMG data quality.

        Parameters
        ----------
        data : np.ndarray
            EMG data
        fs : int
            Sampling frequency

        Returns
        -------
        Dict[str, any]
            Quality metrics
        """
        from scipy import signal

        results = {
            'duration_sec': len(data) / fs,
            'channels': data.shape[1] if len(data.shape) > 1 else 1
        }

        # Check noise level (high frequency content)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        for ch in range(data.shape[1]):
            # Power line noise check (50/60 Hz)
            freqs, psd = signal.welch(data[:, ch], fs, nperseg=min(256, len(data)))

            # Check 60 Hz component
            idx_60 = np.argmin(np.abs(freqs - 60))
            noise_60 = psd[idx_60]
            total_power = np.sum(psd)

            results[f'ch{ch}_noise_ratio'] = noise_60 / total_power
            results[f'ch{ch}_needs_notch'] = noise_60 / total_power > 0.1

        return results

    @staticmethod
    def check_sync_quality(
        imu_onset: int,
        emg_onset: int,
        force_onset: int,
        fs: int = 100
    ) -> Dict[str, any]:
        """
        Assess synchronization quality.

        Parameters
        ----------
        imu_onset : int
            IMU onset sample
        emg_onset : int
            EMG onset sample
        force_onset : int
            Force onset sample
        fs : int
            Common sampling frequency

        Returns
        -------
        Dict[str, any]
            Sync quality metrics
        """
        # Calculate delays in ms
        imu_emg_delay = abs(imu_onset - emg_onset) / fs * 1000
        imu_force_delay = abs(imu_onset - force_onset) / fs * 1000

        # Quality assessment (acceptable < 50ms)
        acceptable_delay = 50  # ms

        return {
            'imu_emg_delay_ms': imu_emg_delay,
            'imu_force_delay_ms': imu_force_delay,
            'sync_quality': 'Good' if max(imu_emg_delay, imu_force_delay) < acceptable_delay else 'Check alignment',
            'max_delay_ms': max(imu_emg_delay, imu_force_delay)
        }


# Pre-collection checklist data
PRECOLLECTION_CHECKLIST = {
    'equipment': [
        'Movella DOT charged (>50%)',
        'DOT firmware updated',
        'OpenBCI battery/connection OK',
        'EMG electrodes within expiry',
        'Kinvent calibrated'
    ],
    'environment': [
        'Magnetic interference sources removed',
        'Consistent lighting',
        'Adequate measurement space'
    ],
    'subject': [
        'Consent form signed',
        'Anthropometric data recorded',
        'Skin prepared (EMG sites)',
        'Sensor placement marked'
    ]
}

POSTCOLLECTION_CHECKLIST = {
    'data_verification': [
        'Files saved successfully',
        'Signal quality reviewed',
        'Backup completed'
    ],
    'documentation': [
        'Collection conditions recorded',
        'Sensor placement photos taken',
        'Subject feedback documented'
    ]
}
