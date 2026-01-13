"""
Biomechanics Analysis Package

Rehabilitation biomechanics analysis tools for:
- Shoulder: Rotator cuff function evaluation
- Lumbar spine: Posture and movement pattern analysis
- Knee: Cartilage load evaluation

Equipment support:
- Movella DOT IMU sensors
- OpenBCI EEG/EMG sensors
- Kinvent force plates
"""

__version__ = "0.1.0"

from .emg_processing import (
    process_emg,
    normalize_emg,
    calculate_mvc,
    calculate_activation_timing,
    calculate_rms,
    calculate_mean_frequency,
    calculate_median_frequency,
    calculate_cocontraction_index,
    remove_ecg_artifact,
    EMGProcessor,
)

from .shoulder_analysis import (
    ShoulderAnalysis,
    ShoulderKinematics,
    analyze_elevation_task,
    analyze_rotator_cuff_test,
)

from .lumbar_analysis import (
    LumbarSpineAnalysis,
    LumbarKinematics,
    DiscLoadEstimate,
    estimate_disc_load,
    classify_lifting_strategy,
    PostureAssessment,
)

from .knee_analysis import (
    KneeAnalysis,
    KneeKinematics,
    MuscleActivation,
    analyze_squat_biomechanics,
    analyze_stair_negotiation,
    calculate_asymmetry_index,
    OALoadManagement,
)

from .data_recorder import (
    MultiModalRecorder,
    RecordingSession,
    synchronize_offline,
    detect_movement_onset,
    detect_emg_onset,
    detect_force_onset,
    DataQualityChecker,
    PRECOLLECTION_CHECKLIST,
    POSTCOLLECTION_CHECKLIST,
)

__all__ = [
    # Version
    "__version__",
    # EMG Processing
    "process_emg",
    "normalize_emg",
    "calculate_mvc",
    "calculate_activation_timing",
    "calculate_rms",
    "calculate_mean_frequency",
    "calculate_median_frequency",
    "calculate_cocontraction_index",
    "remove_ecg_artifact",
    "EMGProcessor",
    # Shoulder Analysis
    "ShoulderAnalysis",
    "ShoulderKinematics",
    "analyze_elevation_task",
    "analyze_rotator_cuff_test",
    # Lumbar Analysis
    "LumbarSpineAnalysis",
    "LumbarKinematics",
    "DiscLoadEstimate",
    "estimate_disc_load",
    "classify_lifting_strategy",
    "PostureAssessment",
    # Knee Analysis
    "KneeAnalysis",
    "KneeKinematics",
    "MuscleActivation",
    "analyze_squat_biomechanics",
    "analyze_stair_negotiation",
    "calculate_asymmetry_index",
    "OALoadManagement",
    # Data Recording
    "MultiModalRecorder",
    "RecordingSession",
    "synchronize_offline",
    "detect_movement_onset",
    "detect_emg_onset",
    "detect_force_onset",
    "DataQualityChecker",
    "PRECOLLECTION_CHECKLIST",
    "POSTCOLLECTION_CHECKLIST",
]
