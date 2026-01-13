"""
Shoulder Biomechanics Analysis Module

Analysis pipeline for rotator cuff function evaluation using:
- Movella DOT IMU sensors (5 sensors: thorax, scapula, humerus, forearm, hand)
- OpenBCI EMG (8 channels: trapezius, deltoid, rotator cuff, etc.)

Based on rehabilitation biomechanics practical guide.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ShoulderKinematics:
    """Container for shoulder kinematics data."""
    humerothoracic: np.ndarray  # Humerus relative to thorax
    scapulothoracic: np.ndarray  # Scapula relative to thorax
    glenohumeral: np.ndarray  # Humerus relative to scapula
    scapular_upward_rotation: np.ndarray
    scapular_protraction: np.ndarray
    scapular_tilt: np.ndarray


class ShoulderAnalysis:
    """
    Shoulder biomechanics analysis for rotator cuff function evaluation.

    Attributes
    ----------
    imu_data : Dict[str, Dict]
        IMU data for each sensor (thorax, scapula, humerus, forearm, hand)
    emg_data : Dict[str, np.ndarray]
        EMG data for each muscle channel
    fs_imu : int
        IMU sampling frequency
    fs_emg : int
        EMG sampling frequency
    """

    # Normal reference values
    NORMAL_SCAPULOHUMERAL_RATIO = 2.0  # 2:1 after 30 deg abduction
    NORMAL_SCAPULAR_UPWARD_ROTATION_MAX = 55  # degrees at full elevation
    NORMAL_UPPER_TRAP_SERRATUS_RATIO = 1.5  # threshold for compensation

    def __init__(
        self,
        imu_data: Dict[str, Dict],
        emg_data: Dict[str, np.ndarray],
        fs_imu: int = 120,
        fs_emg: int = 250
    ):
        """
        Initialize shoulder analysis.

        Parameters
        ----------
        imu_data : Dict[str, Dict]
            IMU data with keys: 'thorax', 'scapula', 'humerus', 'forearm', 'hand'
            Each contains 'pitch', 'roll', 'yaw' or quaternion data
        emg_data : Dict[str, np.ndarray]
            EMG data with muscle names as keys
        fs_imu : int
            IMU sampling frequency (default 120Hz for Movella DOT)
        fs_emg : int
            EMG sampling frequency (default 250Hz for OpenBCI Cyton)
        """
        self.imu = imu_data
        self.emg = emg_data
        self.fs_imu = fs_imu
        self.fs_emg = fs_emg

    def get_segment_angle(
        self,
        segment: str,
        reference: str,
        plane: str = 'sagittal'
    ) -> np.ndarray:
        """
        Calculate relative angle between two segments.

        Parameters
        ----------
        segment : str
            Distal segment name
        reference : str
            Reference segment name
        plane : str
            'sagittal' (pitch), 'frontal' (roll), or 'transverse' (yaw)

        Returns
        -------
        np.ndarray
            Relative angle in degrees
        """
        plane_map = {'sagittal': 'pitch', 'frontal': 'roll', 'transverse': 'yaw'}
        angle_key = plane_map.get(plane, 'pitch')

        segment_angle = self.imu[segment][angle_key]
        reference_angle = self.imu[reference][angle_key]

        return segment_angle - reference_angle

    def calculate_scapulohumeral_rhythm(self) -> Dict[str, np.ndarray]:
        """
        Calculate scapulohumeral rhythm.

        Normal ratio: 2:1 (humerus:scapula) after 30 deg abduction

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'humerothoracic': humerus angle relative to thorax
            - 'scapulothoracic': scapula angle relative to thorax
            - 'glenohumeral': humerus angle relative to scapula
            - 'rhythm_ratio': instantaneous rhythm ratio
        """
        # Humerothoracic angle (total arm elevation)
        humerothoracic = self.get_segment_angle('humerus', 'thorax', 'frontal')

        # Scapulothoracic angle (scapula contribution)
        scapulothoracic = self.get_segment_angle('scapula', 'thorax', 'frontal')

        # Glenohumeral angle (difference)
        glenohumeral = humerothoracic - scapulothoracic

        # Rhythm ratio (per 10-degree intervals)
        rhythm_ratio = np.zeros_like(humerothoracic)
        diff_gh = np.diff(glenohumeral)
        diff_st = np.diff(scapulothoracic)

        # Avoid division by zero
        valid_idx = np.abs(diff_st) > 0.1
        rhythm_ratio[1:][valid_idx] = diff_gh[valid_idx] / diff_st[valid_idx]

        return {
            'humerothoracic': humerothoracic,
            'scapulothoracic': scapulothoracic,
            'glenohumeral': glenohumeral,
            'rhythm_ratio': rhythm_ratio
        }

    def analyze_muscle_balance(self) -> Dict[str, np.ndarray]:
        """
        Analyze rotator cuff vs deltoid muscle balance.

        Normal: Co-activation during abduction
        Abnormal: Deltoid dominance, rotator cuff delay/weakness

        Returns
        -------
        Dict[str, np.ndarray]
            Muscle balance metrics
        """
        # Combined deltoid activity
        deltoid = (
            self.emg.get('anterior_deltoid', np.zeros(1)) +
            self.emg.get('middle_deltoid', np.zeros(1))
        )

        # Combined rotator cuff activity
        rotator_cuff = (
            self.emg.get('supraspinatus', np.zeros(1)) +
            self.emg.get('infraspinatus', np.zeros(1))
        )

        # Balance ratio (normal: ~1.0, RC weakness: <0.5)
        balance_ratio = rotator_cuff / (deltoid + 1e-10)

        return {
            'deltoid_total': deltoid,
            'rotator_cuff_total': rotator_cuff,
            'balance_ratio': balance_ratio,
            'mean_balance': np.mean(balance_ratio)
        }

    def detect_compensation_patterns(self) -> Dict[str, any]:
        """
        Detect compensation patterns during shoulder movement.

        Patterns detected:
        - Excessive scapular elevation (upper trap over-activity)
        - Scapular winging (serratus weakness)

        Returns
        -------
        Dict[str, any]
            Compensation pattern indicators
        """
        upper_trap = self.emg.get('upper_trapezius', np.zeros(1))
        serratus = self.emg.get('serratus_anterior', np.zeros(1))

        # Upper trap / serratus ratio
        # Normal: <1.5, Abnormal: >2.0
        compensation_index = upper_trap / (serratus + 1e-10)
        mean_compensation = np.mean(compensation_index)

        # Scapular kinematics for winging detection
        scapular_tilt = self.get_segment_angle('scapula', 'thorax', 'sagittal')

        return {
            'compensation_index': compensation_index,
            'mean_compensation_index': mean_compensation,
            'scapular_anterior_tilt': scapular_tilt,
            'upper_trap_dominant': mean_compensation > 2.0,
            'possible_serratus_weakness': mean_compensation > 2.5
        }

    def analyze_impingement_risk(
        self,
        abduction_angle: np.ndarray,
        pain_threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Analyze impingement risk during abduction.

        Evaluates EMG changes in 60-120 degree range (painful arc).

        Parameters
        ----------
        abduction_angle : np.ndarray
            Shoulder abduction angle over time
        pain_threshold : float, optional
            Angle at which patient reports pain

        Returns
        -------
        Dict[str, any]
            Impingement risk indicators
        """
        # Define painful arc range
        arc_mask = (abduction_angle >= 60) & (abduction_angle <= 120)

        # EMG in painful arc vs outside
        if np.any(arc_mask):
            emg_in_arc = {}
            emg_outside_arc = {}

            for muscle, data in self.emg.items():
                # Resample EMG to match IMU timeline if needed
                if len(data) != len(abduction_angle):
                    data = np.interp(
                        np.linspace(0, 1, len(abduction_angle)),
                        np.linspace(0, 1, len(data)),
                        data
                    )

                emg_in_arc[muscle] = np.mean(data[arc_mask])
                emg_outside_arc[muscle] = np.mean(data[~arc_mask])

            # Supraspinatus activity change in arc
            supra_change = (
                emg_in_arc.get('supraspinatus', 0) -
                emg_outside_arc.get('supraspinatus', 0)
            )

            return {
                'painful_arc_present': True,
                'emg_in_arc': emg_in_arc,
                'emg_outside_arc': emg_outside_arc,
                'supraspinatus_change': supra_change,
                'impingement_likely': supra_change < -10  # %MVC drop
            }
        else:
            return {
                'painful_arc_present': False,
                'note': 'Abduction did not reach painful arc range'
            }

    def generate_clinical_report(self) -> Dict[str, any]:
        """
        Generate comprehensive clinical interpretation.

        Returns
        -------
        Dict[str, any]
            Clinical findings and recommendations
        """
        rhythm = self.calculate_scapulohumeral_rhythm()
        balance = self.analyze_muscle_balance()
        compensation = self.detect_compensation_patterns()

        findings = []
        recommendations = []

        # Scapulohumeral rhythm assessment
        mean_rhythm = np.mean(rhythm['rhythm_ratio'][rhythm['rhythm_ratio'] > 0])
        if mean_rhythm < 1.5:
            findings.append("Altered scapulohumeral rhythm (early scapular motion)")
            recommendations.append("Scapular stabilization exercises")
        elif mean_rhythm > 2.5:
            findings.append("Delayed scapular contribution")
            recommendations.append("Scapular mobility exercises")

        # Muscle balance assessment
        if balance['mean_balance'] < 0.5:
            findings.append("Rotator cuff weakness relative to deltoid")
            recommendations.append("Rotator cuff strengthening (isometric initially)")

        # Compensation assessment
        if compensation['upper_trap_dominant']:
            findings.append("Upper trapezius dominant pattern")
            recommendations.append("Upper trap relaxation, lower trap activation")

        if compensation['possible_serratus_weakness']:
            findings.append("Possible serratus anterior weakness")
            recommendations.append("Serratus anterior strengthening (wall slides)")

        return {
            'scapulohumeral_rhythm': {
                'ratio': mean_rhythm,
                'normal_range': '1.5-2.5',
                'status': 'Normal' if 1.5 <= mean_rhythm <= 2.5 else 'Abnormal'
            },
            'muscle_balance': {
                'rc_deltoid_ratio': balance['mean_balance'],
                'status': 'Normal' if balance['mean_balance'] > 0.7 else 'Imbalanced'
            },
            'compensation': {
                'index': compensation['mean_compensation_index'],
                'status': 'Normal' if compensation['mean_compensation_index'] < 2.0 else 'Compensating'
            },
            'findings': findings,
            'recommendations': recommendations
        }


def analyze_elevation_task(
    imu_data: Dict[str, Dict],
    emg_data: Dict[str, np.ndarray],
    task_type: str = 'flexion'
) -> Dict[str, any]:
    """
    Analyze shoulder elevation task (flexion or abduction).

    Parameters
    ----------
    imu_data : Dict[str, Dict]
        IMU sensor data
    emg_data : Dict[str, np.ndarray]
        EMG channel data
    task_type : str
        'flexion' or 'abduction'

    Returns
    -------
    Dict[str, any]
        Task analysis results
    """
    analyzer = ShoulderAnalysis(imu_data, emg_data)

    results = {
        'task_type': task_type,
        'rhythm': analyzer.calculate_scapulohumeral_rhythm(),
        'muscle_balance': analyzer.analyze_muscle_balance(),
        'compensation': analyzer.detect_compensation_patterns(),
        'clinical_report': analyzer.generate_clinical_report()
    }

    return results


def analyze_rotator_cuff_test(
    emg_data: Dict[str, np.ndarray],
    test_type: str = 'empty_can'
) -> Dict[str, any]:
    """
    Analyze rotator cuff specific tests.

    Parameters
    ----------
    emg_data : Dict[str, np.ndarray]
        EMG data during test
    test_type : str
        'empty_can', 'external_rotation', 'lift_off'

    Returns
    -------
    Dict[str, any]
        Test results
    """
    test_configs = {
        'empty_can': {
            'target_muscle': 'supraspinatus',
            'normal_activation': 50,  # %MVC
        },
        'external_rotation': {
            'target_muscle': 'infraspinatus',
            'normal_activation': 40,
        },
        'lift_off': {
            'target_muscle': 'subscapularis',
            'normal_activation': 45,
        }
    }

    config = test_configs.get(test_type, test_configs['empty_can'])
    target = config['target_muscle']

    target_activation = np.max(emg_data.get(target, np.zeros(1)))

    # Check for compensation
    deltoid_activation = np.max(
        emg_data.get('anterior_deltoid', np.zeros(1)) +
        emg_data.get('middle_deltoid', np.zeros(1))
    ) / 2

    return {
        'test_type': test_type,
        'target_muscle': target,
        'target_activation': target_activation,
        'expected_activation': config['normal_activation'],
        'deltoid_compensation': deltoid_activation,
        'weakness_indicated': target_activation < config['normal_activation'] * 0.5,
        'compensation_present': deltoid_activation > target_activation
    }
