# Biomechanics Analysis Package

재활의학 생체역학 분석을 위한 Python 패키지입니다.

## 지원 장비

| 장비 | 용도 | 샘플링 |
|------|------|--------|
| **Movella DOT** | IMU 센서 (최대 5개) | 120Hz |
| **OpenBCI Cyton** | EMG/EEG (8ch) | 250Hz |
| **Kinvent** | 포스플레이트 | Bluetooth |

## 설치

```bash
pip install -r requirements.txt
```

## 모듈 구성

### 1. EMG 신호처리 (`emg_processing.py`)

```python
from biomechanics import process_emg, EMGProcessor

# 단일 채널 처리
envelope = process_emg(raw_emg, fs=250)

# 다중 채널 처리
processor = EMGProcessor(fs=250, channel_names=['VMO', 'VL', 'RF'])
processor.set_mvc('VMO', mvc_value)
results = processor.process_multichannel(data, normalize=True)
```

### 2. 어깨 분석 (`shoulder_analysis.py`)

회전근개 기능 평가 및 견갑상완 리듬 분석

```python
from biomechanics import ShoulderAnalysis

analyzer = ShoulderAnalysis(imu_data, emg_data)
rhythm = analyzer.calculate_scapulohumeral_rhythm()
balance = analyzer.analyze_muscle_balance()
report = analyzer.generate_clinical_report()
```

**측정 변수:**
- 견갑상완 리듬 (정상: 2:1)
- 회전근개/삼각근 균형
- 보상 패턴 (상부승모근 과활성)

### 3. 요추 분석 (`lumbar_analysis.py`)

자세 및 동작 패턴 분석, 추간판 부하 추정

```python
from biomechanics import LumbarSpineAnalysis, estimate_disc_load

analyzer = LumbarSpineAnalysis(imu_data, emg_data)
angles = analyzer.calculate_lumbar_angles()
hinge = analyzer.assess_hip_hinge_quality()
core = analyzer.analyze_core_activation()

# 추간판 부하 추정
load = estimate_disc_load(
    lumbar_angle=30,
    external_load=20,
    body_mass=70
)
```

**측정 변수:**
- 요추 전만각, 분절별 각도
- 요추-골반 리듬
- 힙힌지 품질 (요추 중립 유지)
- 코어 동시수축 지수

### 4. 무릎 분석 (`knee_analysis.py`)

연골 부하 평가 및 근활성 패턴 분석

```python
from biomechanics import KneeAnalysis, analyze_squat_biomechanics

analyzer = KneeAnalysis(imu_data, emg_data, force_data, body_weight)
angles = analyzer.calculate_knee_angles()
load = analyzer.estimate_knee_load_proxy()
muscles = analyzer.analyze_muscle_activation()

# 스쿼트 분석
squat = analyze_squat_biomechanics(imu_data, force_data, emg_data, body_weight)
```

**측정 변수:**
- 무릎 굴곡/신전 각도
- Q/H 비율, VMO/VL 비율
- 좌우 비대칭 지수
- 누적 부하

### 5. 데이터 동기화 (`data_recorder.py`)

LSL 기반 다중 장비 동기화

```python
from biomechanics import MultiModalRecorder, synchronize_offline

# 실시간 수집
recorder = MultiModalRecorder(fs_target=100)
recorder.connect_streams()
session = recorder.record(duration_sec=30)
recorder.save_session(session, 'session_001')

# 오프라인 동기화
synced = synchronize_offline(
    imu_file='imu_data.npz',
    emg_file='emg_data.npz',
    force_file='force_data.npz'
)
```

## 영역별 센서 배치

### 어깨 (5 IMU + 8 EMG)
```
IMU: 흉골, 견갑골, 상완, 전완, 손
EMG: 승모근(상/중), 삼각근(전/중), 극상근, 극하근, 대흉근, 전거근
```

### 요추 (4 IMU + 8 EMG)
```
IMU: T12, L3, S1, 대퇴
EMG: 척추기립근(L3 양측), 다열근(L5 양측), 외복사근(양측), 복직근, 대둔근
```

### 무릎 (3 IMU + 8 EMG + 2 Force)
```
IMU: 골반, 대퇴, 경골
EMG: VMO, VL, RF, 내측햄스트링, 외측햄스트링, 비복근, 전경골근, 대둔근
Force: 좌/우 플레이트
```

## 임상 참조값

| 지표 | 정상 범위 | 임상적 의미 |
|------|----------|------------|
| 견갑상완 리듬 | 1.5-2.5:1 | <1.5: 조기 견갑골 운동 |
| 상부승모근/전거근 | <1.5 | >2.0: 보상 패턴 |
| 힙힌지 요추 변화 | <10° | >20°: 과도한 요추 굴곡 |
| 코어 동시수축 | 0.3-0.5 | 과제 의존적 |
| Q/H 비율 | 1.5-2.5 | 근력 균형 |
| VMO/VL 비율 | ~1.0 | <0.8: VMO 약화 |

## 주의사항

- IMU 측정 오차: 동적 측정 시 ±2-5°
- 추간판 부하 추정은 연구용 참고치
- Kinvent는 Fz만 측정 (KAM 직접 계산 불가)

## 라이선스

MIT License
