# 재활의학 생체역학 분석 실무 가이드
## 보유 장비 기반 맞춤 프로토콜

> **보유 장비**  
> - Movella DOT IMU 센서  
> - OpenBCI EEG/EMG 센서  
> - Kinvent 포스플레이트

---

# 장비 특성 및 활용 전략

## 1. Movella DOT IMU 센서

### 스펙 요약
| 항목 | 사양 |
|------|------|
| 센서 수 | 최대 5개 동기화 (Pro: 더 많음) |
| 샘플링 | 최대 120Hz |
| 출력 | 쿼터니언, 오일러각, 가속도, 각속도 |
| 정확도 | Roll/Pitch ±0.5°, Heading ±2° (동적) |
| 연결 | Bluetooth 5.0 |
| SDK | Python, Unity, Android/iOS |

### 강점과 한계
```
[강점]                              [한계]
✓ 휴대성, 임상 환경 적합            ✗ 5개 센서 제한 (동시 측정)
✓ 무선, 환자 움직임 자유            ✗ 자기장 간섭 민감
✓ Python SDK로 자동화 가능          ✗ 드리프트 (장시간 측정)
✓ 실시간 스트리밍                   ✗ 절대 위치 측정 불가
```

### 영역별 센서 배치 전략

**어깨 분석 (4-5센서):**
```
[센서 배치]
    ① 흉골/흉추 (기준)
         │
    ② 견갑골 ─── ③ 상완
                    │
                 ④ 전완
                    │
                 ⑤ 손 (선택)

측정 변수:
- 견갑흉곽 운동학 (upward rotation, protraction, tilt)
- 견관절 굴곡/외전/회전
- 상완-흉곽 각도
```

**요추 분석 (3-4센서):**
```
[센서 배치]
    ① T12/L1 (상부 요추)
         │
    ② L3 (중부 요추)  
         │
    ③ S1/골반 (기준)
         │
    ④ 대퇴 (힙힌지 평가용)

측정 변수:
- 요추 전체 ROM (굴곡/신전/측굴/회전)
- 분절간 상대 각도
- 요추-골반 리듬
- 힙힌지 시 요추 중립 유지 정도
```

**무릎 분석 (3센서):**
```
[센서 배치]
    ① 골반/천골
         │
    ② 대퇴
         │
    ③ 경골

측정 변수:
- 무릎 굴곡/신전 각도
- 보행 주기 중 무릎 운동학
- 기능적 과제 시 관절 각도 프로파일
```

---

## 2. OpenBCI EEG/EMG 센서

### 시스템 구성 (추정: Cyton 또는 Ganglion)
| 보드 | 채널 수 | 샘플링 | EMG 적합성 |
|------|---------|--------|-----------|
| Cyton | 8ch (확장 16ch) | 250Hz | ★★★ 우수 |
| Ganglion | 4ch | 200Hz | ★★☆ 적합 |

### EMG 활용을 위한 설정

**하드웨어 연결:**
```
[OpenBCI + EMG 구성]

전극 타입: Ag/AgCl 표면 전극 (별도 구매 필요)
연결 방식: 
  - 스냅 버튼 전극 + OpenBCI 리드선
  - 또는 터치프루프 커넥터

권장 추가 구매:
  - 의료용 표면 EMG 전극 (3M, Noraxon 등)
  - 도전성 젤
  - 피부 전처리 용품 (알코올, 샌드페이퍼)
```

**영역별 EMG 채널 배치:**

**어깨 (8채널 Cyton 기준):**
| 채널 | 근육 | 역할 | 전극 위치 |
|------|------|------|----------|
| 1 | 상부 승모근 | 견갑골 거상 | 쇄골 중간, 견봉 사이 |
| 2 | 중부 승모근 | 견갑골 내전 | T3 극돌기 높이, 외측 |
| 3 | 전면 삼각근 | 어깨 굴곡 | 견봉 전외측 하방 |
| 4 | 중면 삼각근 | 어깨 외전 | 견봉 외측 하방 |
| 5 | 극상근 | 외전 보조, 관절안정 | 견갑극 상방 |
| 6 | 극하근 | 외회전, 후방안정 | 견갑극 하방 |
| 7 | 대흉근 | 어깨 내전/굴곡 | 쇄골 하방, 내측 |
| 8 | 전거근 | 견갑골 전인/상방회전 | 5-6번 늑골 측면 |

**요추/코어 (8채널):**
| 채널 | 근육 | 역할 | 전극 위치 |
|------|------|------|----------|
| 1-2 | 척추기립근 L3 (양측) | 요추 신전 | L3 극돌기 외측 2-3cm |
| 3-4 | 다열근 L5 (양측) | 분절 안정화 | L5 극돌기 외측 1-2cm |
| 5-6 | 외복사근 (양측) | 회전, 측굴 | 늑골 하연, 외측 |
| 7 | 복직근 | 체간 굴곡 | 배꼽 외측 3cm |
| 8 | 대둔근 | 힙힌지 주동근 | 대전자-천골 중간 |

**무릎 (8채널):**
| 채널 | 근육 | 역할 |
|------|------|------|
| 1 | 내측광근 (VMO) | 슬개골 안정화 |
| 2 | 외측광근 (VL) | 무릎 신전 |
| 3 | 대퇴직근 (RF) | 무릎 신전, 고관절 굴곡 |
| 4 | 내측 햄스트링 | 무릎 굴곡, 내회전 |
| 5 | 외측 햄스트링 | 무릎 굴곡, 외회전 |
| 6 | 비복근 내측두 | 무릎 굴곡, 족저굴곡 |
| 7 | 전경골근 | 족배굴곡 |
| 8 | 대둔근 | 고관절 신전 |

### OpenBCI EMG 신호처리 파이프라인

```python
# EMG 처리 기본 파이프라인 (Python)
import numpy as np
from scipy import signal

def process_emg(raw_emg, fs=250):
    """
    OpenBCI EMG 신호처리
    
    Parameters:
    -----------
    raw_emg : array, 원시 EMG 신호
    fs : int, 샘플링 주파수 (Cyton: 250Hz)
    
    Returns:
    --------
    envelope : array, 선형 엔벨로프
    """
    # 1. 대역통과 필터 (20-450Hz, EMG 대역)
    # OpenBCI는 250Hz이므로 나이퀴스트 = 125Hz
    # 따라서 20-120Hz로 제한
    nyq = fs / 2
    low = 20 / nyq
    high = 120 / nyq  # 나이퀴스트 아래로 제한
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, raw_emg)
    
    # 2. 전파정류
    rectified = np.abs(filtered)
    
    # 3. 저역통과 필터 (선형 엔벨로프, 6Hz cutoff)
    low_env = 6 / nyq
    b_env, a_env = signal.butter(4, low_env, btype='low')
    envelope = signal.filtfilt(b_env, a_env, rectified)
    
    # 4. 정규화 (MVC 대비 %)
    # mvc_value는 별도 MVC 시험에서 획득
    # normalized = (envelope / mvc_value) * 100
    
    return envelope

def calculate_activation_timing(envelope, threshold=0.1):
    """
    근활성 시작/종료 시점 검출
    
    Parameters:
    -----------
    envelope : array, EMG 엔벨로프
    threshold : float, 활성 역치 (최대값 대비 비율)
    
    Returns:
    --------
    onset, offset : 활성 시작/종료 인덱스
    """
    thresh_value = threshold * np.max(envelope)
    active = envelope > thresh_value
    
    # 시작/종료점 찾기
    diff = np.diff(active.astype(int))
    onset = np.where(diff == 1)[0]
    offset = np.where(diff == -1)[0]
    
    return onset, offset
```

---

## 3. Kinvent 포스플레이트

### 스펙 및 특성
| 항목 | K-Force Plates (추정) |
|------|----------------------|
| 측정 | Fz (수직력) |
| 출력 | 힘, COP (센서 배열 시) |
| 크기 | 휴대용, 소형 |
| 연결 | Bluetooth, Kinvent 앱 |
| 동기화 | 2개 플레이트 동시 사용 가능 |

### 활용 전략

**단일 플레이트:**
- 체중 분배 평가
- 한발 서기 균형
- 스쿼트/런지 시 수직력 프로파일

**듀얼 플레이트:**
```
[양발 분리 측정]
┌─────────┐    ┌─────────┐
│  Left   │    │  Right  │
│  Plate  │    │  Plate  │
└─────────┘    └─────────┘

측정 가능:
- 좌우 체중 분배 비율
- 각 하지 수직력 시간 곡선
- 좌우 비대칭 지수
```

### 한계 및 보완 전략
```
[Kinvent 한계]                    [보완 전략]
Fz만 측정 (Fx, Fy 없음)     →    KAM 직접 계산 불가
                                  → 대안: 무릎 각도 + Fz 조합 추정
                                  
소형 크기                    →    보행 분석 제한
                                  → 대안: 정적/준정적 과제 집중
                                  
COP 정밀도 제한              →    상세 균형 분석 한계
                                  → 대안: IMU 자세 동요 보완
```

---

# 영역별 통합 프로토콜

## 영역 1: 어깨 - 회전근개 기능 평가

### 측정 프로토콜

**장비 설정:**
```
Movella DOT (5개): 흉골, 견갑골, 상완, 전완, 손
OpenBCI (8ch): 승모근(상/중), 삼각근(전/중), 극상근, 극하근, 대흉근, 전거근
동기화: LSL (Lab Streaming Layer) 또는 타임스탬프 기반
```

**측정 과제:**
| 과제 | 목적 | 분석 변수 |
|------|------|----------|
| 1. 능동 거상 (굴곡) | 견갑상완 리듬 | 분절 각도 비율, EMG 패턴 |
| 2. 능동 외전 | 충돌 평가 | 90° 부근 EMG 변화, pain arc |
| 3. 등척성 외회전 | 회전근개 기능 | 극하근/삼각근 비율 |
| 4. Empty can test | 극상근 평가 | 극상근 활성도, 보상 패턴 |
| 5. 저항 외전 | Force couple | 삼각근 vs 회전근개 동시수축 |

**데이터 수집 절차:**
```
1. 센서 부착 및 캘리브레이션 (10분)
   - DOT: N-pose 캘리브레이션
   - EMG: MVC 측정 (각 근육 3초 x 3회)

2. 기능적 과제 수행 (20분)
   - 각 과제 5회 반복
   - 30초 휴식 between 과제

3. 임상 검사 동시 수행 (선택)
   - Neer test, Hawkins test 중 EMG 기록
```

### 분석 파이프라인

```python
# 어깨 분석 통합 스크립트 구조

class ShoulderAnalysis:
    def __init__(self, imu_data, emg_data):
        self.imu = imu_data   # Movella DOT
        self.emg = emg_data   # OpenBCI
        
    def calculate_scapulohumeral_rhythm(self):
        """
        견갑상완 리듬 계산
        정상: 2:1 (상완:견갑골) after 30° abduction
        """
        # 흉곽 대비 상완 각도
        humerothoracic = self.get_segment_angle('humerus', 'thorax')
        
        # 흉곽 대비 견갑골 각도  
        scapulothoracic = self.get_segment_angle('scapula', 'thorax')
        
        # 견갑상완 각도 (차이)
        glenohumeral = humerothoracic - scapulothoracic
        
        # 리듬 비율 (매 10° 구간)
        rhythm_ratio = np.diff(glenohumeral) / np.diff(scapulothoracic)
        
        return rhythm_ratio
    
    def analyze_muscle_balance(self):
        """
        회전근개 vs 삼각근 균형 분석
        """
        deltoid = self.emg['anterior_deltoid'] + self.emg['middle_deltoid']
        rotator_cuff = self.emg['supraspinatus'] + self.emg['infraspinatus']
        
        # 정상: 외전 시 동시활성
        # 비정상: 삼각근 우세, 회전근개 지연/약화
        balance_ratio = rotator_cuff / (deltoid + 0.001)
        
        return balance_ratio
    
    def detect_compensation_patterns(self):
        """
        보상 패턴 검출
        - 과도한 견갑골 거상 (상부 승모근 과활성)
        - 견갑골 익상 (전거근 약화)
        """
        upper_trap = self.emg['upper_trapezius']
        serratus = self.emg['serratus_anterior']
        
        # 상부승모근/전거근 비율
        # 정상: <1.5, 비정상: >2.0
        compensation_index = upper_trap / (serratus + 0.001)
        
        return compensation_index
```

### 임상 해석 프레임워크

```
[정상 패턴]
- 견갑상완 리듬: 2:1 (>30° 외전 후)
- 삼각근-회전근개 동시활성
- 상부승모근/전거근 비율 <1.5
- 견갑골 upward rotation 50-60° at full elevation

[회전근개 기능부전 패턴]
- 리듬 변화: 조기 견갑골 운동 (1:1 또는 역전)
- 삼각근 우세: 회전근개 활성 지연/감소
- 보상적 상부승모근 과활성
- 견갑골 anterior tilt 증가 → 충돌 위험 ↑

[관절중심화 실패 징후] (IMU로 간접 추정)
- 외전 60-120° 구간에서 급격한 견갑골 패턴 변화
- 통증 호소 구간과 운동학적 이상 일치 여부
```

---

## 영역 2: 요추 - 자세/동작 패턴 분석

### 측정 프로토콜

**장비 설정:**
```
Movella DOT (4개): T12, L3, S1, 대퇴
OpenBCI (8ch): 척추기립근(L3 양측), 다열근(L5 양측), 
               외복사근(양측), 복직근, 대둔근
Kinvent (선택): 스쿼트/데드리프트 시 수직력
```

**측정 과제:**
| 과제 | 목적 | 핵심 변수 |
|------|------|----------|
| 1. 선 자세 정적 평가 | 기본 정렬 | 요추 전만각 |
| 2. 최대 굴곡-신전 | ROM, 리듬 | 요추-골반 기여도 |
| 3. 앉았다 일어서기 | 기능적 패턴 | 요추 각도 변화, 근활성 |
| 4. 힙힌지 패턴 | 부하 분산 전략 | 요추 중립 유지 정도 |
| 5. 물건 들어올리기 | 실제 부하 과제 | 전략 (squat vs stoop) |
| 6. 코어 브레이싱 | 안정화 능력 | 동시수축 패턴 |

**데이터 수집 절차:**
```
1. 센서 부착 (10분)
   - DOT: 극돌기에 양면테이프 부착
   - EMG: SENIAM 가이드라인 따름

2. MVC 측정 (10분)
   - 척추기립근: prone extension
   - 복근: curl-up
   - 대둔근: prone hip extension

3. 과제 수행 (25분)
   - 각 과제 5회 반복
   - 힙힌지/들어올리기: 무부하 → 점진적 부하
```

### 분석 파이프라인

```python
class LumbarSpineAnalysis:
    def __init__(self, imu_data, emg_data, force_data=None):
        self.imu = imu_data
        self.emg = emg_data
        self.force = force_data
        
    def calculate_lumbar_angles(self):
        """
        요추 각도 계산 (L1-S1)
        """
        # T12 센서와 S1 센서 간 상대 각도
        l1_angle = self.imu['T12']['pitch']  # sagittal plane
        s1_angle = self.imu['S1']['pitch']
        
        lumbar_lordosis = l1_angle - s1_angle
        
        # L3 센서로 상부/하부 요추 구분
        upper_lumbar = self.imu['T12']['pitch'] - self.imu['L3']['pitch']
        lower_lumbar = self.imu['L3']['pitch'] - self.imu['S1']['pitch']
        
        return {
            'total_lordosis': lumbar_lordosis,
            'upper_lumbar': upper_lumbar,
            'lower_lumbar': lower_lumbar
        }
    
    def calculate_lumbar_pelvic_rhythm(self):
        """
        요추-골반 리듬 계산
        전방 굴곡 시:
        - 초기: 요추 굴곡 우세
        - 후기: 골반 전경(고관절 굴곡) 우세
        """
        lumbar_angle = self.imu['L3']['pitch'] - self.imu['S1']['pitch']
        pelvic_angle = self.imu['S1']['pitch'] - self.imu['thigh']['pitch']
        
        # 리듬 비율 계산
        total_motion = lumbar_angle + pelvic_angle
        lumbar_contribution = lumbar_angle / (total_motion + 0.001)
        pelvic_contribution = pelvic_angle / (total_motion + 0.001)
        
        return lumbar_contribution, pelvic_contribution
    
    def assess_hip_hinge_quality(self, task_data):
        """
        힙힌지 품질 평가
        
        목표: 요추 중립 유지하며 고관절에서 굴곡
        """
        lumbar_change = np.max(task_data['lumbar']) - np.min(task_data['lumbar'])
        hip_change = np.max(task_data['hip']) - np.min(task_data['hip'])
        
        # 이상적: 요추 변화 <10°, 고관절 변화 >60°
        hinge_quality = hip_change / (lumbar_change + 1)
        
        # 등급화
        if lumbar_change < 10 and hip_change > 60:
            quality = 'Excellent'
        elif lumbar_change < 15 and hip_change > 50:
            quality = 'Good'
        elif lumbar_change < 20:
            quality = 'Fair'
        else:
            quality = 'Poor - Excessive lumbar flexion'
            
        return quality, lumbar_change, hip_change
    
    def analyze_core_activation(self):
        """
        코어 근활성 패턴 분석
        """
        # 굴근/신근 동시수축 지수
        flexors = (self.emg['rectus_abdominis'] + 
                   self.emg['external_oblique_L'] + 
                   self.emg['external_oblique_R']) / 3
        
        extensors = (self.emg['erector_spinae_L'] + 
                     self.emg['erector_spinae_R'] +
                     self.emg['multifidus_L'] +
                     self.emg['multifidus_R']) / 4
        
        cocontraction_index = (2 * np.minimum(flexors, extensors)) / (flexors + extensors + 0.001)
        
        # 좌우 대칭성
        left_side = (self.emg['erector_spinae_L'] + self.emg['multifidus_L']) / 2
        right_side = (self.emg['erector_spinae_R'] + self.emg['multifidus_R']) / 2
        symmetry_index = np.minimum(left_side, right_side) / (np.maximum(left_side, right_side) + 0.001)
        
        return cocontraction_index, symmetry_index
```

### 추간판 부하 추정 모델

```python
def estimate_disc_load(lumbar_angle, external_load, body_mass, 
                       erector_activity, abdominal_activity):
    """
    간이 추간판 부하 추정 모델
    
    기반: McGill의 요추 모델 단순화 버전
    주의: 연구용 추정치, 절대값 해석 주의
    
    Parameters:
    -----------
    lumbar_angle : float, 요추 굴곡 각도 (°)
    external_load : float, 외부 부하 (kg)
    body_mass : float, 체중 (kg)
    erector_activity : float, 척추기립근 EMG (%MVC)
    abdominal_activity : float, 복근 EMG (%MVC)
    """
    import numpy as np
    
    # 상수 (문헌 기반 추정치)
    g = 9.81  # m/s²
    trunk_mass_ratio = 0.45  # 상체 질량 비율
    moment_arm_trunk = 0.15  # 체간 무게중심 모멘트암 (m)
    moment_arm_load = 0.35   # 외부 부하 모멘트암 (m)
    moment_arm_erector = 0.05  # 척추기립근 모멘트암 (m)
    
    # 체간 질량에 의한 모멘트
    trunk_moment = (body_mass * trunk_mass_ratio * g * 
                    moment_arm_trunk * np.sin(np.radians(lumbar_angle)))
    
    # 외부 부하에 의한 모멘트
    load_moment = external_load * g * moment_arm_load
    
    # 총 굴곡 모멘트
    total_flexion_moment = trunk_moment + load_moment
    
    # 척추기립근 힘 추정 (모멘트 균형)
    erector_force = total_flexion_moment / moment_arm_erector
    
    # 추간판 압박력 추정
    # 압박력 = 척추기립근 힘 + 체중 성분 + 부하
    compression = (erector_force + 
                   body_mass * trunk_mass_ratio * g * np.cos(np.radians(lumbar_angle)) +
                   external_load * g)
    
    # IAP 효과 보정 (복근 활성 시 감소)
    iap_reduction = abdominal_activity * 0.1  # 10% 감소 가정
    compression_adjusted = compression * (1 - iap_reduction)
    
    # 전단력 추정
    shear = (body_mass * trunk_mass_ratio * g * 
             np.sin(np.radians(lumbar_angle)))
    
    return {
        'compression_N': compression_adjusted,
        'compression_BW': compression_adjusted / (body_mass * g),
        'shear_N': shear,
        'flexion_moment_Nm': total_flexion_moment
    }
```

### 임상 해석 프레임워크

```
[자세별 추간판 부하 등급]

낮은 부하 (<1.0 BW 압박):
- 누운 자세
- 지지된 앉기 (등받이 100-110°)

중간 부하 (1.0-2.0 BW):
- 서기 (중립 정렬)
- 지지 없는 앉기
- 힙힌지 들어올리기 (경량)

높은 부하 (2.0-3.0 BW):
- 요추 굴곡 자세
- 무거운 물건 들어올리기
- 굴곡 + 회전

위험 부하 (>3.0 BW):
- 급격한 굴곡 + 중량
- 굴곡 + 회전 + 중량
- NIOSH 권장 압박 한계: 3400N ≈ 4-5 BW

[코어 안정화 패턴]

정상:
- 동시수축 지수: 0.3-0.5 (과제 의존적)
- 좌우 대칭성: >0.8
- 사전 활성화 (부하 전 50ms 이내)

비정상:
- 동시수축 부재 또는 과도 (>0.7)
- 비대칭 >20%
- 지연된 활성화
- 특정 근육 억제 (예: 다열근)
```

---

## 영역 3: 무릎 - 연골 부하 평가

### 측정 프로토콜

**장비 설정:**
```
Movella DOT (3개): 골반, 대퇴, 경골
OpenBCI (8ch): VMO, VL, RF, 내측햄스트링, 외측햄스트링, 
               비복근, 전경골근, 대둔근
Kinvent (2개): 양발 분리 수직력 측정
```

**측정 과제:**
| 과제 | 목적 | 핵심 변수 |
|------|------|----------|
| 1. 정적 서기 | 체중 분배 | 좌우 비율 |
| 2. 스쿼트 | 기능적 부하 | 무릎 각도-수직력 관계 |
| 3. 한발 스쿼트 | 단일하지 기능 | 무릎 각도, 불안정성 |
| 4. 계단 오르기/내려가기 | 일상 부하 | 굴곡각, 수직력 패턴 |
| 5. 앉았다 일어서기 (STS) | 기능 평가 | 전략, 속도, 대칭성 |
| 6. 등속성 신전/굴곡 | 근력 평가 | Q/H 비율 |

**수정된 KAM 추정 접근:**
```
[문제] Kinvent는 Fz만 측정 → 전통적 KAM 계산 불가

[대안적 접근]
1. 무릎 내반 각도 + Fz 조합
   - 내반 모멘트 ∝ Fz × 무릎 내반각 × 레버암
   
2. 동적 내반 각도 (IMU 측정)
   - 대퇴-경골 frontal plane 각도
   - 보행/스쿼트 중 변화 추적
   
3. Proxy 지표 사용
   - 내측/외측 근활성 비율 (VMO/VL)
   - 체중 분배 비대칭
```

### 분석 파이프라인

```python
class KneeAnalysis:
    def __init__(self, imu_data, emg_data, force_data):
        self.imu = imu_data
        self.emg = emg_data
        self.force = force_data
        
    def calculate_knee_angles(self):
        """
        무릎 관절 각도 계산
        """
        # 대퇴-경골 상대 각도
        thigh_orientation = self.imu['thigh']
        shank_orientation = self.imu['shank']
        
        # 시상면 (굴곡/신전)
        knee_flexion = thigh_orientation['pitch'] - shank_orientation['pitch']
        
        # 관상면 (내반/외반) - 주의: IMU 정확도 제한
        knee_varus = thigh_orientation['roll'] - shank_orientation['roll']
        
        return knee_flexion, knee_varus
    
    def estimate_knee_load_proxy(self):
        """
        무릎 부하 대리 지표 계산
        
        Kinvent 한계로 인해 직접 KAM 계산 불가
        대안적 부하 추정 지표 사용
        """
        # 1. 수직력 기반 부하 (체중 정규화)
        fz_left = self.force['left'] / self.body_weight
        fz_right = self.force['right'] / self.body_weight
        
        # 2. 내반 각도 × 수직력 (단순화된 모멘트 추정)
        knee_varus_L, knee_varus_R = self.calculate_frontal_angles()
        
        # 모멘트 프록시 (단위 없음, 상대 비교용)
        moment_proxy_L = fz_left * np.sin(np.radians(knee_varus_L))
        moment_proxy_R = fz_right * np.sin(np.radians(knee_varus_R))
        
        # 3. 비대칭 지수
        asymmetry = (fz_right - fz_left) / (fz_right + fz_left) * 100
        
        return {
            'vertical_force_L': fz_left,
            'vertical_force_R': fz_right,
            'moment_proxy_L': moment_proxy_L,
            'moment_proxy_R': moment_proxy_R,
            'asymmetry_percent': asymmetry
        }
    
    def analyze_muscle_activation(self):
        """
        무릎 근활성 패턴 분석
        """
        # Q/H 비율 (대퇴사두근/햄스트링)
        quadriceps = (self.emg['VMO'] + self.emg['VL'] + self.emg['RF']) / 3
        hamstrings = (self.emg['med_ham'] + self.emg['lat_ham']) / 2
        
        qh_ratio = quadriceps / (hamstrings + 0.001)
        
        # VMO/VL 비율 (슬개골 안정성)
        vmo_vl_ratio = self.emg['VMO'] / (self.emg['VL'] + 0.001)
        
        # 내측/외측 햄스트링 비율
        ham_ratio = self.emg['med_ham'] / (self.emg['lat_ham'] + 0.001)
        
        return {
            'QH_ratio': qh_ratio,
            'VMO_VL_ratio': vmo_vl_ratio,
            'hamstring_ratio': ham_ratio
        }
    
    def calculate_cumulative_load(self, task_duration, repetitions):
        """
        누적 부하 계산
        
        연골 퇴행은 순간 부하보다 누적 부하와 관련
        """
        # 단일 과제 부하 적분
        single_task_load = np.trapz(self.force['total'], dx=1/self.sampling_rate)
        
        # 일일/주간 누적 추정
        daily_repetitions = repetitions  # 예: 계단 오르기 횟수
        daily_cumulative = single_task_load * daily_repetitions
        
        return {
            'single_task_impulse': single_task_load,
            'daily_cumulative': daily_cumulative,
            'weekly_cumulative': daily_cumulative * 7
        }
```

### 스쿼트 분석 상세

```python
def analyze_squat_biomechanics(imu_data, force_data, emg_data):
    """
    스쿼트 생체역학 종합 분석
    
    임상 적용: OA 환자 운동 처방 근거
    """
    results = {}
    
    # 1. 운동학적 분석
    knee_flexion = calculate_knee_flexion(imu_data)
    hip_flexion = calculate_hip_flexion(imu_data)
    trunk_lean = calculate_trunk_angle(imu_data)
    
    results['kinematics'] = {
        'max_knee_flexion': np.max(knee_flexion),
        'max_hip_flexion': np.max(hip_flexion),
        'max_trunk_lean': np.max(trunk_lean),
        'knee_hip_ratio': np.max(knee_flexion) / np.max(hip_flexion)
    }
    
    # 2. 수직력 분석
    peak_force = np.max(force_data['total'])
    force_at_max_flexion = force_data['total'][np.argmax(knee_flexion)]
    
    results['kinetics'] = {
        'peak_vertical_force_BW': peak_force,
        'force_at_max_flexion_BW': force_at_max_flexion,
        'loading_rate': calculate_loading_rate(force_data)
    }
    
    # 3. 근활성 분석
    results['muscle_activation'] = {
        'peak_quad': np.max(emg_data['quadriceps']),
        'peak_hamstring': np.max(emg_data['hamstrings']),
        'quad_onset': detect_onset(emg_data['quadriceps']),
        'cocontraction': calculate_cocontraction(emg_data)
    }
    
    # 4. 부하 등급화
    # OA 환자용 안전 기준
    if results['kinematics']['max_knee_flexion'] > 90:
        flexion_risk = 'High'
    elif results['kinematics']['max_knee_flexion'] > 60:
        flexion_risk = 'Moderate'
    else:
        flexion_risk = 'Low'
        
    results['clinical_interpretation'] = {
        'flexion_risk': flexion_risk,
        'recommended_depth': 'Partial (<60°)' if flexion_risk == 'High' else 'Full'
    }
    
    return results
```

### 임상 해석 프레임워크

```
[무릎 부하 위험 요인]

해부학적:
- 내반슬 (varus) → 내측 구획 과부하
- 외반슬 (valgus) → 외측 구획, PF 과부하
- Q각 증가 → 슬개대퇴 스트레스

운동학적:
- 보행 시 trunk sway → KAM 증가
- 스쿼트 시 무릎 전방 이동 과다 → PF 부하
- 계단 내려가기 충격 흡수 부전

근육 불균형:
- VMO 약화 → 슬개골 외측 추적
- 햄스트링 약화 → ACL 부담, 과신전
- 대둔근 약화 → 무릎 내반 모멘트 증가

[중재 전략 - 생체역학 기반]

1. 부하 감소
   - 체중 감량: 1kg ↓ → 무릎 부하 4kg ↓
   - 보행 속도 조절
   - 충격 흡수 신발/깔창

2. 부하 재분배  
   - lateral wedge insole (내반슬)
   - 무릎 보조기 (unloader brace)
   - 보행 패턴 수정 (trunk lean)

3. 근력 강화
   - 대퇴사두근 (특히 VMO)
   - 고관절 외전근 (gluteus medius)
   - 햄스트링 균형

4. 운동학적 수정
   - 스쿼트 깊이 제한 (OA)
   - 계단 오르기 전략 (step-over-step vs step-by-step)
```

---

# 통합 데이터 수집 시스템

## 동기화 전략

### Lab Streaming Layer (LSL) 기반 동기화

```python
"""
LSL을 이용한 다중 장비 동기화
- Movella DOT: LSL 출력 지원 (SDK 설정 필요)
- OpenBCI: LSL 지원 (OpenBCI GUI 또는 Brainflow)
- Kinvent: 타임스탬프 기반 후처리 동기화
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
from datetime import datetime

class MultiModalRecorder:
    def __init__(self):
        self.streams = {}
        self.data = {}
        
    def connect_streams(self):
        """LSL 스트림 연결"""
        # IMU 스트림 찾기
        print("Looking for IMU stream...")
        imu_streams = resolve_stream('type', 'IMU')
        if imu_streams:
            self.streams['imu'] = StreamInlet(imu_streams[0])
            
        # EMG 스트림 찾기
        print("Looking for EMG stream...")
        emg_streams = resolve_stream('type', 'EMG')
        if emg_streams:
            self.streams['emg'] = StreamInlet(emg_streams[0])
            
    def record(self, duration_sec):
        """동기화된 데이터 수집"""
        start_time = datetime.now()
        
        self.data = {
            'imu': [],
            'emg': [],
            'timestamps': []
        }
        
        while (datetime.now() - start_time).total_seconds() < duration_sec:
            # IMU 데이터
            if 'imu' in self.streams:
                sample, timestamp = self.streams['imu'].pull_sample(timeout=0.0)
                if sample:
                    self.data['imu'].append(sample)
                    self.data['timestamps'].append(timestamp)
                    
            # EMG 데이터  
            if 'emg' in self.streams:
                sample, timestamp = self.streams['emg'].pull_sample(timeout=0.0)
                if sample:
                    self.data['emg'].append(sample)
                    
        return self.data
    
    def save_data(self, filename):
        """데이터 저장"""
        np.savez(filename, 
                 imu=np.array(self.data['imu']),
                 emg=np.array(self.data['emg']),
                 timestamps=np.array(self.data['timestamps']))
```

### 대안: 타임스탬프 기반 후처리 동기화

```python
def synchronize_offline(imu_file, emg_file, force_file, sync_event='movement_onset'):
    """
    오프라인 동기화 (각 장비 개별 수집 시)
    
    동기화 이벤트:
    - 급격한 움직임 시작 (IMU 가속도 피크)
    - 힘 플레이트 접촉 (수직력 급증)
    - EMG 버스트 시작
    """
    # 각 데이터 로드
    imu_data = load_imu_data(imu_file)
    emg_data = load_emg_data(emg_file)
    force_data = load_force_data(force_file)
    
    # 동기화 이벤트 검출
    imu_sync = detect_movement_onset(imu_data['acceleration'])
    emg_sync = detect_emg_burst(emg_data)
    force_sync = detect_force_onset(force_data)
    
    # 시간 오프셋 계산
    offset_emg = imu_sync - emg_sync
    offset_force = imu_sync - force_sync
    
    # 데이터 정렬
    emg_aligned = shift_data(emg_data, offset_emg)
    force_aligned = shift_data(force_data, offset_force)
    
    # 공통 샘플링 레이트로 리샘플링
    target_fs = 100  # Hz
    imu_resampled = resample_data(imu_data, target_fs)
    emg_resampled = resample_data(emg_aligned, target_fs)
    force_resampled = resample_data(force_aligned, target_fs)
    
    return {
        'imu': imu_resampled,
        'emg': emg_resampled,
        'force': force_resampled,
        'sync_quality': assess_sync_quality(imu_sync, emg_sync, force_sync)
    }
```

---

# 품질 관리 체크리스트

## 데이터 수집 전

```markdown
## 장비 점검
- [ ] Movella DOT 충전 상태 (>50%)
- [ ] DOT 펌웨어 업데이트 확인
- [ ] OpenBCI 배터리/연결 상태
- [ ] EMG 전극 유효기간 확인
- [ ] Kinvent 캘리브레이션 상태

## 환경 설정
- [ ] 자기장 간섭원 제거 (DOT 정확도)
- [ ] 조명 일정 (일부 센서 영향)
- [ ] 측정 공간 확보

## 피험자 준비
- [ ] 동의서 서명
- [ ] 인체측정 데이터 기록
- [ ] 피부 전처리 (EMG 부위)
- [ ] 센서 부착 위치 마킹
```

## 데이터 수집 중

```markdown
## 실시간 모니터링
- [ ] IMU 신호 연속성 확인
- [ ] EMG 노이즈 레벨 점검
- [ ] 포스플레이트 영점 확인

## 프로토콜 준수
- [ ] 과제 순서 기록
- [ ] 휴식 시간 준수
- [ ] 특이사항 메모
```

## 데이터 수집 후

```markdown
## 데이터 검증
- [ ] 파일 저장 확인
- [ ] 신호 품질 1차 검토
- [ ] 백업 완료

## 문서화
- [ ] 수집 조건 기록
- [ ] 센서 배치 사진
- [ ] 피험자 피드백 기록
```

---

# 12개월 실행 계획 (수정)

## Phase 1: 시스템 구축 (1-3개월)

### 월별 목표

**1개월차:**
- [ ] Movella DOT SDK 설치 및 Python 연동
- [ ] 5개 센서 동시 수집 테스트
- [ ] 기본 운동학 계산 스크립트 개발

**2개월차:**
- [ ] OpenBCI EMG 설정 최적화
- [ ] 8채널 EMG 수집 및 처리 파이프라인
- [ ] MVC 측정 프로토콜 확립

**3개월차:**
- [ ] Kinvent 연동 및 동기화 테스트
- [ ] 3개 장비 통합 수집 프로토콜
- [ ] 파일럿 데이터 수집 (자체 테스트)

## Phase 2: 프로토콜 검증 (4-6개월)

**4개월차:**
- [ ] 어깨 프로토콜 정상군 5명 수집
- [ ] 견갑상완 리듬 분석 검증

**5개월차:**
- [ ] 요추 프로토콜 정상군 5명 수집
- [ ] 힙힌지 평가 알고리즘 검증

**6개월차:**
- [ ] 무릎 프로토콜 정상군 5명 수집
- [ ] 스쿼트 분석 알고리즘 검증

## Phase 3: 정상 데이터베이스 (7-9개월)

- [ ] 영역당 정상군 15-20명 수집
- [ ] 연령/성별별 정상 참조값 산출
- [ ] 자동화 분석 파이프라인 완성

## Phase 4: 임상 연구 (10-12개월)

- [ ] IRB 승인 (환자 연구)
- [ ] 환자군 데이터 수집 시작
- [ ] 정상 vs 환자 예비 비교
- [ ] 학회 발표 초록 준비

---

# 예상 한계 및 보완 전략

## 장비별 한계

| 장비 | 한계 | 보완 전략 |
|------|------|----------|
| **Movella DOT** | 5센서 제한, 드리프트 | 필수 분절 우선, 짧은 시행 |
| | 절대 위치 불가 | 관절 각도 중심 분석 |
| | 자기장 간섭 | 환경 통제, 보정 알고리즘 |
| **OpenBCI** | 연구용 EMG 대비 정밀도 | 상대 비교 중심 해석 |
| | 움직임 아티팩트 | 필터링, 전극 고정 강화 |
| | 8채널 제한 | 핵심 근육 선별, 세션 분리 |
| **Kinvent** | Fz만 측정 | 대리 지표 활용 |
| | 소형 크기 | 정적/준정적 과제 집중 |
| | 보행 분석 제한 | 추후 보행 분석 장비 고려 |

## 연구 설계 시 고려사항

```
1. 측정 오차 인정
   - IMU: 동적 측정 ±2-5° 오차
   - 상대적 변화량 비교에 집중
   - 절대값보다 패턴 분석

2. 타당도 확보
   - 가능 시 영상 분석과 비교 검증
   - 기존 문헌값과 비교
   - 테스트-재테스트 신뢰도 평가

3. 임상적 유의성
   - 통계적 유의성 + 임상적 의미
   - Minimal Clinically Important Difference (MCID) 고려
   - 환자 기능/증상과의 상관 분석
```

---

*이 장비 구성으로 충분히 의미 있는 임상 연구가 가능합니다.*
*광학식 모션캡처 없이도 IMU 기반 운동학 분석은 학술적으로 인정받는 방법입니다.*
*점진적으로 경험을 쌓으며 필요시 장비를 확장하시면 됩니다.*
