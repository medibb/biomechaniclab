"""
Lumbar Spine Biomechanics Analysis Module

Analysis pipeline for posture and movement pattern analysis using:
- Movella DOT IMU sensors (4 sensors: T12, L3, S1, thigh)
- OpenBCI EMG (8 channels: erector spinae, multifidus, obliques, etc.)
- Kinvent force plates (optional)

Based on rehabilitation biomechanics practical guide.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LumbarKinematics:
    """Container for lumbar kinematics data."""
    total_lordosis: np.ndarray
    upper_lumbar: np.ndarray  # T12-L3
    lower_lumbar: np.ndarray  # L3-S1
    pelvic_tilt: np.ndarray
    hip_angle: np.ndarray


@dataclass
class DiscLoadEstimate:
    """Container for disc load estimates."""
    compression_N: float
    compression_BW: float
    shear_N: float
    flexion_moment_Nm: float


class LumbarSpineAnalysis:
    """
    Lumbar spine biomechanics analysis for posture and movement patterns.

    Attributes
    ----------
    imu_data : Dict[str, Dict]
        IMU data for each sensor (T12, L3, S1, thigh)
    emg_data : Dict[str, np.ndarray]
        EMG data for each muscle channel
    force_data : Dict[str, np.ndarray], optional
        Force plate data
    fs_imu : int
        IMU sampling frequency
    fs_emg : int
        EMG sampling frequency
    """

    # Risk thresholds based on NIOSH guidelines
    NIOSH_COMPRESSION_LIMIT = 3400  # N
    SAFE_LUMBAR_FLEXION_LIFTING = 10  # degrees

    def __init__(
        self,
        imu_data: Dict[str, Dict],
        emg_data: Dict[str, np.ndarray],
        force_data: Optional[Dict[str, np.ndarray]] = None,
        fs_imu: int = 120,
        fs_emg: int = 250
    ):
        """
        Initialize lumbar spine analysis.

        Parameters
        ----------
        imu_data : Dict[str, Dict]
            IMU data with keys: 'T12', 'L3', 'S1', 'thigh'
        emg_data : Dict[str, np.ndarray]
            EMG data with muscle names as keys
        force_data : Dict[str, np.ndarray], optional
            Force plate data
        fs_imu : int
            IMU sampling frequency
        fs_emg : int
            EMG sampling frequency
        """
        self.imu = imu_data
        self.emg = emg_data
        self.force = force_data
        self.fs_imu = fs_imu
        self.fs_emg = fs_emg

    def calculate_lumbar_angles(self) -> Dict[str, np.ndarray]:
        """
        Calculate lumbar spine angles.

        Returns
        -------
        Dict[str, np.ndarray]
            Lumbar angle components
        """
        # T12 and S1 sensor sagittal angles
        t12_pitch = self.imu['T12']['pitch']
        l3_pitch = self.imu['L3']['pitch']
        s1_pitch = self.imu['S1']['pitch']

        # Total lumbar lordosis (T12-S1)
        total_lordosis = t12_pitch - s1_pitch

        # Upper lumbar (T12-L3)
        upper_lumbar = t12_pitch - l3_pitch

        # Lower lumbar (L3-S1)
        lower_lumbar = l3_pitch - s1_pitch

        return {
            'total_lordosis': total_lordosis,
            'upper_lumbar': upper_lumbar,
            'lower_lumbar': lower_lumbar
        }

    def calculate_lumbar_pelvic_rhythm(self) -> Dict[str, np.ndarray]:
        """
        Calculate lumbar-pelvic rhythm during forward bending.

        Normal pattern:
        - Initial: Lumbar flexion dominant
        - Late: Pelvic anterior tilt (hip flexion) dominant

        Returns
        -------
        Dict[str, np.ndarray]
            Rhythm components and contributions
        """
        # Lumbar angle (L3-S1)
        lumbar_angle = self.imu['L3']['pitch'] - self.imu['S1']['pitch']

        # Pelvic/hip angle (S1-thigh)
        pelvic_angle = self.imu['S1']['pitch'] - self.imu['thigh']['pitch']

        # Total forward bend
        total_motion = np.abs(lumbar_angle) + np.abs(pelvic_angle)

        # Avoid division by zero
        total_motion = np.maximum(total_motion, 1e-10)

        # Contribution ratios
        lumbar_contribution = np.abs(lumbar_angle) / total_motion * 100
        pelvic_contribution = np.abs(pelvic_angle) / total_motion * 100

        return {
            'lumbar_angle': lumbar_angle,
            'pelvic_angle': pelvic_angle,
            'lumbar_contribution_percent': lumbar_contribution,
            'pelvic_contribution_percent': pelvic_contribution,
            'total_flexion': total_motion
        }

    def assess_hip_hinge_quality(
        self,
        task_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Assess hip hinge movement quality.

        Goal: Maintain lumbar neutral while flexing at hips

        Parameters
        ----------
        task_data : Dict[str, np.ndarray], optional
            Task-specific data. If None, uses stored IMU data.

        Returns
        -------
        Dict[str, any]
            Hip hinge quality assessment
        """
        if task_data is None:
            lumbar = self.imu['L3']['pitch'] - self.imu['S1']['pitch']
            hip = self.imu['S1']['pitch'] - self.imu['thigh']['pitch']
        else:
            lumbar = task_data.get('lumbar', np.zeros(1))
            hip = task_data.get('hip', np.zeros(1))

        lumbar_change = np.max(lumbar) - np.min(lumbar)
        hip_change = np.max(hip) - np.min(hip)

        # Quality grading
        # Ideal: lumbar change <10 deg, hip change >60 deg
        if lumbar_change < 10 and hip_change > 60:
            quality = 'Excellent'
            score = 4
        elif lumbar_change < 15 and hip_change > 50:
            quality = 'Good'
            score = 3
        elif lumbar_change < 20:
            quality = 'Fair'
            score = 2
        else:
            quality = 'Poor - Excessive lumbar flexion'
            score = 1

        # Calculate hinge ratio
        hinge_ratio = hip_change / (lumbar_change + 1)

        return {
            'quality': quality,
            'score': score,
            'lumbar_excursion_deg': lumbar_change,
            'hip_excursion_deg': hip_change,
            'hinge_ratio': hinge_ratio,
            'target_lumbar': '<10 degrees',
            'target_hip': '>60 degrees'
        }

    def analyze_core_activation(self) -> Dict[str, np.ndarray]:
        """
        Analyze core muscle activation patterns.

        Returns
        -------
        Dict[str, np.ndarray]
            Core activation metrics
        """
        # Flexor group
        rectus = self.emg.get('rectus_abdominis', np.zeros(1))
        oblique_l = self.emg.get('external_oblique_L', np.zeros(1))
        oblique_r = self.emg.get('external_oblique_R', np.zeros(1))
        flexors = (rectus + oblique_l + oblique_r) / 3

        # Extensor group
        es_l = self.emg.get('erector_spinae_L', np.zeros(1))
        es_r = self.emg.get('erector_spinae_R', np.zeros(1))
        mf_l = self.emg.get('multifidus_L', np.zeros(1))
        mf_r = self.emg.get('multifidus_R', np.zeros(1))
        extensors = (es_l + es_r + mf_l + mf_r) / 4

        # Co-contraction index
        # Formula: 2 * min(flexors, extensors) / (flexors + extensors)
        min_activity = np.minimum(flexors, extensors)
        cocontraction = (2 * min_activity) / (flexors + extensors + 1e-10)

        # Left-right symmetry for extensors
        left_ext = (es_l + mf_l) / 2
        right_ext = (es_r + mf_r) / 2
        symmetry = np.minimum(left_ext, right_ext) / (
            np.maximum(left_ext, right_ext) + 1e-10
        )

        return {
            'flexor_activity': flexors,
            'extensor_activity': extensors,
            'cocontraction_index': cocontraction,
            'mean_cocontraction': np.mean(cocontraction),
            'symmetry_index': symmetry,
            'mean_symmetry': np.mean(symmetry)
        }

    def detect_multifidus_inhibition(self) -> Dict[str, any]:
        """
        Detect multifidus inhibition pattern (common in LBP).

        Returns
        -------
        Dict[str, any]
            Multifidus assessment
        """
        mf_l = self.emg.get('multifidus_L', np.zeros(1))
        mf_r = self.emg.get('multifidus_R', np.zeros(1))
        es_l = self.emg.get('erector_spinae_L', np.zeros(1))
        es_r = self.emg.get('erector_spinae_R', np.zeros(1))

        # Multifidus to erector spinae ratio
        # Low ratio suggests multifidus inhibition
        mf_es_ratio_l = np.mean(mf_l) / (np.mean(es_l) + 1e-10)
        mf_es_ratio_r = np.mean(mf_r) / (np.mean(es_r) + 1e-10)

        # Side difference
        mf_asymmetry = np.abs(np.mean(mf_l) - np.mean(mf_r)) / (
            np.maximum(np.mean(mf_l), np.mean(mf_r)) + 1e-10
        ) * 100

        inhibition_suspected = (mf_es_ratio_l < 0.5) or (mf_es_ratio_r < 0.5)

        return {
            'mf_es_ratio_left': mf_es_ratio_l,
            'mf_es_ratio_right': mf_es_ratio_r,
            'asymmetry_percent': mf_asymmetry,
            'inhibition_suspected': inhibition_suspected,
            'affected_side': 'left' if mf_es_ratio_l < mf_es_ratio_r else 'right'
        }


def estimate_disc_load(
    lumbar_angle: float,
    external_load: float,
    body_mass: float,
    erector_activity: float = 0.5,
    abdominal_activity: float = 0.3
) -> Dict[str, float]:
    """
    Estimate intervertebral disc load (simplified McGill model).

    WARNING: Research estimate only, not for absolute clinical decisions.

    Parameters
    ----------
    lumbar_angle : float
        Lumbar flexion angle (degrees)
    external_load : float
        External load (kg)
    body_mass : float
        Body mass (kg)
    erector_activity : float
        Erector spinae EMG (0-1 normalized)
    abdominal_activity : float
        Abdominal EMG (0-1 normalized)

    Returns
    -------
    Dict[str, float]
        Estimated disc loads
    """
    g = 9.81  # m/s^2

    # Model constants (literature-based estimates)
    trunk_mass_ratio = 0.45  # Upper body mass ratio
    moment_arm_trunk = 0.15  # Trunk CoM moment arm (m)
    moment_arm_load = 0.35   # External load moment arm (m)
    moment_arm_erector = 0.05  # Erector spinae moment arm (m)

    # Convert angle to radians
    angle_rad = np.radians(lumbar_angle)

    # Trunk moment (body weight)
    trunk_mass = body_mass * trunk_mass_ratio
    trunk_moment = trunk_mass * g * moment_arm_trunk * np.sin(angle_rad)

    # External load moment
    load_moment = external_load * g * moment_arm_load

    # Total flexion moment
    total_flexion_moment = trunk_moment + load_moment

    # Erector force required to balance moment
    erector_force = total_flexion_moment / moment_arm_erector

    # Compression force estimation
    compression = (
        erector_force +
        trunk_mass * g * np.cos(angle_rad) +
        external_load * g
    )

    # IAP effect (abdominal bracing reduces compression)
    iap_reduction = abdominal_activity * 0.1  # 10% max reduction
    compression_adjusted = compression * (1 - iap_reduction)

    # Shear force
    shear = trunk_mass * g * np.sin(angle_rad)

    return {
        'compression_N': compression_adjusted,
        'compression_BW': compression_adjusted / (body_mass * g),
        'shear_N': shear,
        'flexion_moment_Nm': total_flexion_moment,
        'risk_level': 'High' if compression_adjusted > 3400 else 'Acceptable'
    }


def classify_lifting_strategy(
    lumbar_excursion: float,
    hip_excursion: float,
    knee_excursion: float
) -> Dict[str, any]:
    """
    Classify lifting strategy (squat vs stoop vs semi-squat).

    Parameters
    ----------
    lumbar_excursion : float
        Lumbar flexion ROM during lift (degrees)
    hip_excursion : float
        Hip flexion ROM during lift (degrees)
    knee_excursion : float
        Knee flexion ROM during lift (degrees)

    Returns
    -------
    Dict[str, any]
        Strategy classification and recommendations
    """
    if knee_excursion > 90 and lumbar_excursion < 20:
        strategy = 'Squat lift'
        spine_risk = 'Low'
        knee_load = 'High'
    elif knee_excursion < 30 and lumbar_excursion > 40:
        strategy = 'Stoop lift'
        spine_risk = 'High'
        knee_load = 'Low'
    else:
        strategy = 'Semi-squat lift'
        spine_risk = 'Moderate'
        knee_load = 'Moderate'

    # Recommendations based on patient profile
    recommendations = {
        'Squat lift': 'Appropriate for spine protection if knees are healthy',
        'Stoop lift': 'Consider hip hinge training to reduce lumbar flexion',
        'Semi-squat lift': 'Good balance if performed with hip hinge technique'
    }

    return {
        'strategy': strategy,
        'spine_risk': spine_risk,
        'knee_load': knee_load,
        'lumbar_excursion': lumbar_excursion,
        'hip_excursion': hip_excursion,
        'knee_excursion': knee_excursion,
        'recommendation': recommendations[strategy]
    }


class PostureAssessment:
    """Static posture assessment from IMU data."""

    NORMAL_LORDOSIS_RANGE = (30, 50)  # degrees

    def __init__(self, imu_data: Dict[str, Dict]):
        """
        Initialize posture assessment.

        Parameters
        ----------
        imu_data : Dict[str, Dict]
            IMU data in standing position
        """
        self.imu = imu_data

    def assess_standing_lordosis(self) -> Dict[str, any]:
        """
        Assess standing lumbar lordosis.

        Returns
        -------
        Dict[str, any]
            Lordosis assessment
        """
        t12_pitch = np.mean(self.imu['T12']['pitch'])
        s1_pitch = np.mean(self.imu['S1']['pitch'])

        lordosis = t12_pitch - s1_pitch

        if lordosis < self.NORMAL_LORDOSIS_RANGE[0]:
            classification = 'Hypolordotic (flat back)'
        elif lordosis > self.NORMAL_LORDOSIS_RANGE[1]:
            classification = 'Hyperlordotic (excessive curve)'
        else:
            classification = 'Normal lordosis'

        return {
            'lordosis_degrees': lordosis,
            'classification': classification,
            'normal_range': f'{self.NORMAL_LORDOSIS_RANGE[0]}-{self.NORMAL_LORDOSIS_RANGE[1]} degrees'
        }

    def assess_pelvic_position(self) -> Dict[str, any]:
        """
        Assess pelvic tilt in standing.

        Returns
        -------
        Dict[str, any]
            Pelvic position assessment
        """
        s1_pitch = np.mean(self.imu['S1']['pitch'])

        # Approximate pelvic tilt from S1 angle
        # Note: True ASIS-PSIS angle requires anatomical landmarks
        if s1_pitch > 15:
            tilt = 'Anterior pelvic tilt'
        elif s1_pitch < 5:
            tilt = 'Posterior pelvic tilt'
        else:
            tilt = 'Neutral pelvis'

        return {
            's1_angle': s1_pitch,
            'tilt_classification': tilt
        }
