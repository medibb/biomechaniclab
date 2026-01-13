"""
Knee Biomechanics Analysis Module

Analysis pipeline for cartilage load evaluation using:
- Movella DOT IMU sensors (3 sensors: pelvis, thigh, shank)
- OpenBCI EMG (8 channels: quadriceps, hamstrings, gastrocnemius, etc.)
- Kinvent force plates (2 plates for bilateral measurement)

Based on rehabilitation biomechanics practical guide.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class KneeKinematics:
    """Container for knee kinematics data."""
    flexion_extension: np.ndarray
    varus_valgus: np.ndarray  # Note: IMU accuracy limited for this
    internal_external_rotation: np.ndarray


@dataclass
class MuscleActivation:
    """Container for knee muscle activation data."""
    quadriceps: np.ndarray
    hamstrings: np.ndarray
    gastrocnemius: np.ndarray
    qh_ratio: np.ndarray


class KneeAnalysis:
    """
    Knee biomechanics analysis for cartilage load evaluation.

    Attributes
    ----------
    imu_data : Dict[str, Dict]
        IMU data for each sensor (pelvis, thigh, shank)
    emg_data : Dict[str, np.ndarray]
        EMG data for each muscle channel
    force_data : Dict[str, np.ndarray]
        Force plate data (left, right)
    body_weight : float
        Body weight in N for normalization
    fs_imu : int
        IMU sampling frequency
    fs_emg : int
        EMG sampling frequency
    """

    # Clinical thresholds
    NORMAL_QH_RATIO_RANGE = (1.5, 2.5)
    NORMAL_VMO_VL_RATIO = 1.0
    ACCEPTABLE_ASYMMETRY = 10  # percent

    def __init__(
        self,
        imu_data: Dict[str, Dict],
        emg_data: Dict[str, np.ndarray],
        force_data: Dict[str, np.ndarray],
        body_weight: float,
        fs_imu: int = 120,
        fs_emg: int = 250
    ):
        """
        Initialize knee analysis.

        Parameters
        ----------
        imu_data : Dict[str, Dict]
            IMU data with keys: 'pelvis', 'thigh', 'shank'
        emg_data : Dict[str, np.ndarray]
            EMG data with muscle names as keys
        force_data : Dict[str, np.ndarray]
            Force plate data with 'left' and 'right' keys
        body_weight : float
            Body weight in Newtons
        fs_imu : int
            IMU sampling frequency
        fs_emg : int
            EMG sampling frequency
        """
        self.imu = imu_data
        self.emg = emg_data
        self.force = force_data
        self.body_weight = body_weight
        self.fs_imu = fs_imu
        self.fs_emg = fs_emg

    def calculate_knee_angles(self) -> Dict[str, np.ndarray]:
        """
        Calculate knee joint angles from IMU data.

        Returns
        -------
        Dict[str, np.ndarray]
            Knee angles in three planes
        """
        thigh = self.imu['thigh']
        shank = self.imu['shank']

        # Sagittal plane (flexion/extension)
        knee_flexion = thigh['pitch'] - shank['pitch']

        # Frontal plane (varus/valgus) - Note: IMU accuracy limited
        knee_varus = thigh['roll'] - shank['roll']

        # Transverse plane (rotation)
        knee_rotation = thigh['yaw'] - shank['yaw']

        return {
            'flexion': knee_flexion,
            'varus_valgus': knee_varus,
            'rotation': knee_rotation
        }

    def calculate_frontal_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frontal plane knee angles (both sides).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Left and right knee varus angles
        """
        # For bilateral analysis, would need separate sensor sets
        # This is a placeholder for single-side measurement
        knee_varus = self.imu['thigh']['roll'] - self.imu['shank']['roll']

        return knee_varus, knee_varus  # Placeholder for bilateral

    def estimate_knee_load_proxy(self) -> Dict[str, any]:
        """
        Estimate knee load using proxy measures.

        Note: Direct KAM calculation not possible with Kinvent (Fz only).
        Uses alternative indicators.

        Returns
        -------
        Dict[str, any]
            Knee load proxy measures
        """
        # Vertical force normalized to body weight
        fz_left = self.force.get('left', np.zeros(1)) / self.body_weight
        fz_right = self.force.get('right', np.zeros(1)) / self.body_weight

        # Get frontal plane angles
        knee_varus_L, knee_varus_R = self.calculate_frontal_angles()

        # Simplified moment proxy (Fz x sin(varus angle))
        # Not true KAM, but indicates loading tendency
        moment_proxy_L = fz_left * np.sin(np.radians(knee_varus_L))
        moment_proxy_R = fz_right * np.sin(np.radians(knee_varus_R))

        # Asymmetry index
        total_force = fz_left + fz_right
        asymmetry = np.zeros_like(total_force)
        valid = total_force > 0.1
        asymmetry[valid] = (
            (fz_right[valid] - fz_left[valid]) / total_force[valid] * 100
        )

        return {
            'vertical_force_L_BW': fz_left,
            'vertical_force_R_BW': fz_right,
            'moment_proxy_L': moment_proxy_L,
            'moment_proxy_R': moment_proxy_R,
            'asymmetry_percent': asymmetry,
            'mean_asymmetry': np.mean(np.abs(asymmetry)),
            'peak_force_L': np.max(fz_left),
            'peak_force_R': np.max(fz_right)
        }

    def analyze_muscle_activation(self) -> Dict[str, any]:
        """
        Analyze knee muscle activation patterns.

        Returns
        -------
        Dict[str, any]
            Muscle activation metrics
        """
        # Quadriceps components
        vmo = self.emg.get('VMO', np.zeros(1))
        vl = self.emg.get('VL', np.zeros(1))
        rf = self.emg.get('RF', np.zeros(1))
        quadriceps = (vmo + vl + rf) / 3

        # Hamstrings
        med_ham = self.emg.get('medial_hamstring', np.zeros(1))
        lat_ham = self.emg.get('lateral_hamstring', np.zeros(1))
        hamstrings = (med_ham + lat_ham) / 2

        # Q/H ratio
        qh_ratio = quadriceps / (hamstrings + 1e-10)

        # VMO/VL ratio (patellar stability indicator)
        vmo_vl_ratio = vmo / (vl + 1e-10)

        # Hamstring medial/lateral ratio
        ham_ratio = med_ham / (lat_ham + 1e-10)

        return {
            'quadriceps': quadriceps,
            'hamstrings': hamstrings,
            'QH_ratio': qh_ratio,
            'mean_QH_ratio': np.mean(qh_ratio),
            'VMO_VL_ratio': vmo_vl_ratio,
            'mean_VMO_VL': np.mean(vmo_vl_ratio),
            'hamstring_ratio': ham_ratio,
            'mean_ham_ratio': np.mean(ham_ratio)
        }

    def calculate_cumulative_load(
        self,
        task_duration: float,
        daily_repetitions: int
    ) -> Dict[str, float]:
        """
        Calculate cumulative joint load.

        Cartilage degeneration correlates with cumulative rather than
        instantaneous load.

        Parameters
        ----------
        task_duration : float
            Duration of single task in seconds
        daily_repetitions : int
            Estimated daily repetitions

        Returns
        -------
        Dict[str, float]
            Cumulative load estimates
        """
        total_force = (
            self.force.get('left', np.zeros(1)) +
            self.force.get('right', np.zeros(1))
        )

        # Single task load impulse (force x time integral)
        dt = 1 / self.fs_imu
        single_task_impulse = np.trapz(total_force, dx=dt)

        # Daily cumulative
        daily_cumulative = single_task_impulse * daily_repetitions

        return {
            'single_task_impulse_Ns': single_task_impulse,
            'daily_cumulative_Ns': daily_cumulative,
            'weekly_cumulative_Ns': daily_cumulative * 7,
            'normalized_daily_BW_s': daily_cumulative / self.body_weight
        }

    def assess_patellofemoral_risk(self) -> Dict[str, any]:
        """
        Assess patellofemoral joint stress risk factors.

        Returns
        -------
        Dict[str, any]
            PF risk assessment
        """
        knee_angles = self.calculate_knee_angles()
        max_flexion = np.max(knee_angles['flexion'])

        muscle = self.analyze_muscle_activation()
        vmo_vl = muscle['mean_VMO_VL']

        risk_factors = []

        # Deep flexion increases PF contact stress
        if max_flexion > 90:
            risk_factors.append('Deep knee flexion (>90 deg)')

        # VMO weakness increases lateral patellar tracking
        if vmo_vl < 0.8:
            risk_factors.append('VMO weakness (VMO/VL < 0.8)')

        # Dynamic valgus
        valgus = knee_angles['varus_valgus']
        if np.min(valgus) < -10:  # negative = valgus
            risk_factors.append('Dynamic knee valgus')

        return {
            'max_flexion': max_flexion,
            'vmo_vl_ratio': vmo_vl,
            'min_varus_valgus': np.min(valgus),
            'risk_factors': risk_factors,
            'risk_level': 'High' if len(risk_factors) >= 2 else 'Low'
        }


def analyze_squat_biomechanics(
    imu_data: Dict[str, Dict],
    force_data: Dict[str, np.ndarray],
    emg_data: Dict[str, np.ndarray],
    body_weight: float
) -> Dict[str, any]:
    """
    Comprehensive squat biomechanics analysis.

    Clinical application: Exercise prescription for OA patients.

    Parameters
    ----------
    imu_data : Dict[str, Dict]
        IMU sensor data
    force_data : Dict[str, np.ndarray]
        Force plate data
    emg_data : Dict[str, np.ndarray]
        EMG data
    body_weight : float
        Body weight in N

    Returns
    -------
    Dict[str, any]
        Squat analysis results
    """
    results = {}

    # 1. Kinematics
    knee_flexion = imu_data['thigh']['pitch'] - imu_data['shank']['pitch']
    hip_flexion = imu_data['pelvis']['pitch'] - imu_data['thigh']['pitch']

    # Trunk angle (if available)
    trunk_lean = imu_data.get('trunk', {}).get(
        'pitch',
        imu_data['pelvis']['pitch']
    )

    results['kinematics'] = {
        'max_knee_flexion': np.max(knee_flexion),
        'max_hip_flexion': np.max(hip_flexion),
        'max_trunk_lean': np.max(trunk_lean),
        'knee_hip_ratio': np.max(knee_flexion) / (np.max(hip_flexion) + 1)
    }

    # 2. Vertical force analysis
    total_force = (
        force_data.get('left', np.zeros(1)) +
        force_data.get('right', np.zeros(1))
    )
    peak_force = np.max(total_force) / body_weight

    # Force at maximum knee flexion
    max_flex_idx = np.argmax(knee_flexion)
    force_at_max_flex = total_force[min(max_flex_idx, len(total_force)-1)]

    # Loading rate
    dt = 0.01  # assume 100Hz
    force_diff = np.diff(total_force)
    loading_rate = np.max(force_diff) / dt

    results['kinetics'] = {
        'peak_vertical_force_BW': peak_force,
        'force_at_max_flexion_BW': force_at_max_flex / body_weight,
        'loading_rate_N_per_s': loading_rate
    }

    # 3. Muscle activation
    quad = (
        emg_data.get('VMO', np.zeros(1)) +
        emg_data.get('VL', np.zeros(1)) +
        emg_data.get('RF', np.zeros(1))
    ) / 3

    ham = (
        emg_data.get('medial_hamstring', np.zeros(1)) +
        emg_data.get('lateral_hamstring', np.zeros(1))
    ) / 2

    results['muscle_activation'] = {
        'peak_quadriceps': np.max(quad),
        'peak_hamstrings': np.max(ham),
        'mean_QH_ratio': np.mean(quad) / (np.mean(ham) + 1e-10)
    }

    # 4. Clinical interpretation for OA patients
    max_flex = results['kinematics']['max_knee_flexion']

    if max_flex > 90:
        flexion_risk = 'High'
        recommended_depth = 'Partial squat (<60 deg)'
    elif max_flex > 60:
        flexion_risk = 'Moderate'
        recommended_depth = 'Quarter squat or wall sit'
    else:
        flexion_risk = 'Low'
        recommended_depth = 'Current depth acceptable'

    results['clinical_interpretation'] = {
        'flexion_risk': flexion_risk,
        'recommended_depth': recommended_depth,
        'pf_contact_stress': 'Increases significantly >60 deg flexion',
        'tibiofemoral_load': 'Peak at deepest position'
    }

    return results


def analyze_stair_negotiation(
    imu_data: Dict[str, Dict],
    force_data: Dict[str, np.ndarray],
    direction: str = 'descent'
) -> Dict[str, any]:
    """
    Analyze stair climbing/descending biomechanics.

    Parameters
    ----------
    imu_data : Dict[str, Dict]
        IMU sensor data
    force_data : Dict[str, np.ndarray]
        Force plate data
    direction : str
        'ascent' or 'descent'

    Returns
    -------
    Dict[str, any]
        Stair analysis results
    """
    knee_flexion = imu_data['thigh']['pitch'] - imu_data['shank']['pitch']

    total_force = (
        force_data.get('left', np.zeros(1)) +
        force_data.get('right', np.zeros(1))
    )

    # Peak force timing relative to step cycle
    peak_force_idx = np.argmax(total_force)
    peak_flexion_at_peak_force = knee_flexion[
        min(peak_force_idx, len(knee_flexion)-1)
    ]

    # Loading rate (important for shock absorption)
    dt = 0.01
    loading_rate = np.max(np.diff(total_force)) / dt

    results = {
        'direction': direction,
        'peak_knee_flexion': np.max(knee_flexion),
        'flexion_at_peak_force': peak_flexion_at_peak_force,
        'peak_loading_rate': loading_rate,
        'eccentric_control': 'Required' if direction == 'descent' else 'Minimal'
    }

    # Descent-specific concerns
    if direction == 'descent':
        results['clinical_notes'] = [
            'Eccentric quadriceps control critical',
            'Higher PF joint stress than ascent',
            'Consider step-by-step strategy if painful',
            'Handrail use reduces knee load by ~25%'
        ]
    else:
        results['clinical_notes'] = [
            'Concentric quadriceps demand',
            'Hip extensor contribution important',
            'Consider leading with stronger leg'
        ]

    return results


def calculate_asymmetry_index(
    left_value: np.ndarray,
    right_value: np.ndarray
) -> np.ndarray:
    """
    Calculate limb symmetry index.

    Parameters
    ----------
    left_value : np.ndarray
        Left side measurement
    right_value : np.ndarray
        Right side measurement

    Returns
    -------
    np.ndarray
        Asymmetry index (positive = right dominant)
    """
    total = left_value + right_value
    asymmetry = np.zeros_like(total)

    valid = total > 1e-10
    asymmetry[valid] = (right_value[valid] - left_value[valid]) / total[valid] * 100

    return asymmetry


class OALoadManagement:
    """
    Load management recommendations for knee OA patients.
    """

    DAILY_STEP_RECOMMENDATIONS = {
        'mild_oa': (6000, 8000),
        'moderate_oa': (4000, 6000),
        'severe_oa': (2000, 4000)
    }

    @staticmethod
    def estimate_daily_knee_cycles(
        steps_per_day: int,
        stairs_per_day: int = 20
    ) -> Dict[str, int]:
        """
        Estimate daily knee loading cycles.

        Parameters
        ----------
        steps_per_day : int
            Daily step count
        stairs_per_day : int
            Daily stair steps

        Returns
        -------
        Dict[str, int]
            Loading cycle estimates
        """
        return {
            'walking_cycles': steps_per_day,
            'stair_cycles': stairs_per_day,
            'high_load_cycles': stairs_per_day,  # Stairs = high load
            'total_cycles': steps_per_day + stairs_per_day
        }

    @staticmethod
    def load_modification_strategies() -> Dict[str, List[str]]:
        """
        Return evidence-based load modification strategies.

        Returns
        -------
        Dict[str, List[str]]
            Strategies by category
        """
        return {
            'load_reduction': [
                'Weight loss: 1kg body weight = 4kg knee load reduction',
                'Walking speed reduction',
                'Shock-absorbing footwear/insoles',
                'Assistive device (cane reduces load ~25%)'
            ],
            'load_redistribution': [
                'Lateral wedge insoles for medial OA',
                'Unloader knee brace',
                'Gait retraining (trunk lean)',
                'Toe-out gait pattern'
            ],
            'muscle_strengthening': [
                'Quadriceps (especially VMO)',
                'Hip abductors (gluteus medius)',
                'Hamstring balance',
                'Focus on eccentric control'
            ],
            'activity_modification': [
                'Limit deep squatting',
                'Step-over-step vs step-by-step stairs',
                'Avoid prolonged kneeling',
                'Rest breaks during activity'
            ]
        }
