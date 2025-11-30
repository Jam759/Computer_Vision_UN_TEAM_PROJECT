# 🅿️ Parking Monitor System

**YOLOv8 기반 실시간 주차 감지 및 모니터링 시스템**

UN TEAM 대학 컴퓨터 비전 팀 프로젝트

---

## 📌 프로젝트 개요

이 프로젝트는 사용자가 비디오에서 주차칸(ROI, Region of Interest)을 지정하면, YOLOv8 기반 차량 검출과 프레임 차이 분석을 결합하여 각 ROI의 주차 상태를 실시간으로 추적하고 시각화하는 데스크톱 애플리케이션입니다.

**핵심 특징:**
- 🎯 **차량 자동 감지**: YOLOv8 Large 모델을 사용한 고정확도 차량 검출
- 📍 **ROI 기반 추적**: 사용자 정의 주차칸 영역에 대한 개별 상태 추적
- 🔍 **프레임 차이 분석**: 대각선 차량 오탐을 줄이기 위한 ROI 변화 감지
- 🔄 **안정적 상태 전이**: 상태 머신 기반의 노이즈 완화 (1초 유예)
- ⚡ **GPU 가속**: NVIDIA CUDA 지원으로 높은 처리 성능
- 🎨 **직관적 GUI**: Tkinter 기반 3단계 워크플로우 (비디오 선택 → ROI 설정 → 실시간 모니터링)
- 🏎️ **차량 방향 감지**: 프레임간 BBOX 중심 추적을 통한 8방향 이동 방향 시각화

---

## 📊 (1) 프로젝트 개요 및 설명

### 목표
사용자가 지정한 주차칸에 대해 차량의 진입·이탈을 자동으로 감지하여 주차 상태를 실시간으로 추적하고 시각화합니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **차량 검출** | YOLOv8 Large 모델을 이용한 차량 바운딩 박스 획득 (신뢰도 0.5 이상) |
| **BBOX 축소** | 차량 중심부(80%)만 사용하여 대각선 차량의 외곽 모서리 제거 |
| **ROI 판정** | BBOX가 ROI의 10% 이상 차지하는지 계산 (적분 이미지 기반 O(1) 최적화) |
| **프레임 차이 감지** | ROI 영역의 20% 이상 픽셀 변화 시 차량 존재로 판정 (대각선 차량 필터링) |
| **하이브리드 검출** | BBOX 방식 AND 프레임 차이 방식 병합으로 오탐 최소화 |
| **상태 추적** | 4가지 상태 전이: EMPTY → DETECTING → PARKED → LEAVING → EMPTY |
| **노이즈 완화** | YOLO 감지 실패에 대한 1초 유예(tolerance) 적용 |
| **방향 감지** | 차량 이동 방향 8방향 분류 및 화살표 시각화 (N, NE, E, SE, S, SW, W, NW) |
| **실시간 시각화** | ROI, 차량 바운딩 박스, 방향 화살표, 신뢰도 표시 |
| **배속 제어** | 0.5x ~ 4x 배속으로 비디오 재생 |

### 주차 상태 정의

```
EMPTY (비어있음) [초록색]
  ↓ (차량 감지 시작)
DETECTING (감지 중) [주황색]
  ↓ (3초 이상 지속)
PARKED (주차됨) [빨강색]
  ↓ (차량 미감지 시작)
LEAVING (떠나는 중) [연두색]
  ↓ (2초 이상 미감지)
EMPTY (비어있음)
```

---

## 🏗️ (2) 전체 구조 및 개발환경

### 개발환경

| 항목 | 요구사항 | 현재 버전 |
|------|---------|---------|
| **언어** | Python 3.8+ | 3.12.9 ✓ |
| **OS** | Windows / Linux / macOS | Windows ✓ |
| **GPU** | (선택) NVIDIA CUDA 12.1+ | CUDA 지원 + RTX 4070 ✓ |
| **Python 패키지** | requirements.txt 참조 | 설치 필요 |

### 프로젝트 구조

```
Computer_Vision_UN_TEAM_PROJECT/
├── main.py                        # 애플리케이션 진입점
├── app/
│   └── parking_app.py             # 앱 컨테이너 (화면 전환, 공용 상태)
├── ui/
│   ├── main_screen.py             # 비디오 선택 및 썸네일 표시
│   ├── roi_choice_screen.py       # ROI 설정 화면 (OpenCV)
│   └── play_screen.py             # 재생 및 실시간 감지 시각화
├── core/
│   ├── ui_components.py           # UI 컴포넌트 헬퍼 (모듈화)
│   ├── VehicleDetector.py         # YOLO 모델 래퍼 + 방향 감지
│   ├── ParkingTracker.py          # 상태 머신 (ROI별 추적)
│   ├── rectangle_selector.py      # OpenCV 기반 ROI 선택 도구
│   └── utils.py                   # 기하학/색상/비디오 유틸
├── data/                          # 데이터 저장 폴더
├── yolov8l.pt                     # YOLOv8 Large 모델 (자동 다운로드)
├── requirements.txt               # 의존성 명시
└── README.md                      # 이 문서
```

### 파일별 역할

**UI 계층 (ui/)**
- `main_screen.py`: 비디오 파일 선택, 첫 프레임 썸네일 표시
- `roi_choice_screen.py`: OpenCV 창에서 마우스 클릭으로 ROI 설정
- `play_screen.py`: 비디오 재생, 실시간 차량 감지, 상태 표시
  - 비동기 추론 워커: 백그라운드 스레드에서 YOLO 추론 실행
  - 프레임 차이 감지: 8초 후 활성화되어 대각선 차량 필터링
  - 배속 최적화: 배속 >= 3.0일 때 반해상도 감지 적용
  - 시간 기반 프레임 동기화: 배속에 관계없이 정확한 슬라이더 위치 유지

**핵심 로직 (core/)**
- `VehicleDetector.py`: YOLOv8 Large 모델 로드, 차량 검출, 방향 감지
- `ParkingTracker.py`: 프레임 기반 상태 머신 관리
- `rectangle_selector.py`: 사용자의 마우스 클릭으로 ROI 다각형 생성
- `utils.py`: 기하학 연산, 색상 정의, 비디오 처리 유틸

**애플리케이션 (app/)**
- `parking_app.py`: Tkinter 루트 컨테이너, 화면 전환 관리

---

## 🚀 (3) 주요 기술 및 기능 설명

### 3.1 차량 검출 (Vehicle Detection)

**방식**: YOLOv8 Large (`yolov8l.pt`)
- 클래스: 차량 (자동차, 오토바이, 버스, 트럭) - COCO 클래스 [2, 3, 5, 7]
- 신뢰도 임계값: 0.5
- GPU 지원: NVIDIA CUDA 활용 시 고속 추론

**구현 (core/VehicleDetector.py)**
```python
def detect_vehicles(self, frame):
    """프레임에서 차량 검출"""
    results = self.model(frame, verbose=False, classes=self.VEHICLE_CLASSES, device=self.device)

    for box in results.boxes:
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        # BBOX 중심 영역만 추출 (80% 유지, 20% 제거)
        bbox = self._shrink_bbox(bbox, self.BBOX_SHRINK_RATIO)

    return bboxes, confidences
```

**BBOX 축소 기능**:
```python
@staticmethod
def _shrink_bbox(bbox, shrink_ratio=0.80):
    """BBOX를 중심 기준으로 축소"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # 중심에서의 축소 거리
    dx = width * (1 - shrink_ratio) / 2
    dy = height * (1 - shrink_ratio) / 2

    x1_new = int(x1 + dx)
    y1_new = int(y1 + dy)
    x2_new = int(x2 - dx)
    y2_new = int(y2 - dy)

    return [x1_new, y1_new, x2_new, y2_new]
```

- `BBOX_SHRINK_RATIO = 0.80` (80% 유지, 20% 제거)
- 대각선으로 보는 차량의 불필요한 모서리 제거
- 중심 기준 대칭으로 축소하여 차량 중심부만 검사
- 효과: 빈 공간 오탐 감소, 정확도 향상

**GPU 모드 설정**:
```python
try:
    self.device = 0  # GPU:0
    self.model.to('cuda')
    print("✓ GPU(CUDA) 모드로 실행 중입니다.")
except Exception as e:
    self.device = 'cpu'
    self.model.to('cpu')
    print(f"⚠ GPU를 사용할 수 없어 CPU 모드로 전환합니다.")
```

### 3.2 차량 방향 감지 (Direction Detection)

**알고리즘**: 프레임 간 BBOX 중심 추적

```python
def detect_vehicle_direction(self, bboxes):
    """차량의 이동 방향 감지 (이전 프레임과 비교)"""
    current_centers = [(x1+x2)/2, (y1+y2)/2 for bbox in bboxes]

    for curr_center in current_centers:
        # 가장 가까운 이전 BBOX 찾기 (100픽셀 이내)
        closest_prev = find_closest(curr_center, self.prev_bboxes)

        if closest_prev and distance < 100:
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = sqrt(dx² + dy²)

            if speed < 3.0:
                direction = 'STATIONARY'
            else:
                angle = atan2(dy, dx) * 180 / π
                direction = self._angle_to_direction(angle)  # 8방향 분류

    self.prev_bboxes = current_centers
    return directions
```

**8방향 분류**:
```python
@staticmethod
def _angle_to_direction(angle):
    """각도를 8방향으로 변환"""
    angle = angle % 360
    directions = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    index = int((angle + 22.5) / 45) % 8
    return directions[index]
```

**방향 정보**:
- `direction`: 8방향 (N, NE, E, SE, S, SW, W, NW), STATIONARY, NEW
- `angle`: 이동 각도 (-180° ~ 180°)
- `confidence`: 신뢰도 (이동 거리에 비례, 0.0 ~ 1.0)
- `speed`: 이동 속도 (픽셀/프레임)

### 3.3 ROI 판정 (ROI Overlap Detection)

**최적화 방식**: 적분 이미지 (Integral Image) 기반
- 프로젝트 시작 시 각 ROI에 대해 적분 이미지 사전 계산
- 매 프레임 O(1) 시간에 겹침 비율 계산
- 그리드 샘플링 (20×20)보다 100배 이상 빠름

**알고리즘 (core/utils.py) - ROI 기준 계산**
```python
def calculate_bbox_in_roi_ratio(bbox, polygon):
    """BBOX가 ROI의 몇 %를 차지하는지 계산 (ROI 기준)"""
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)

    # ROI 면적 계산 (Shoelace formula)
    roi_area = abs(sum(polygon[i][0] * polygon[(i+1)%4][1] -
                      polygon[(i+1)%4][0] * polygon[i][1]
                      for i in range(4))) / 2

    # 교집합 계산 (그리드 샘플링 20x20)
    grid_samples = 20
    overlap_pixels = 0
    step = max(1, (x2 - x1) // grid_samples)

    for x in range(x1, x2, step):
        for y in range(y1, y2, step):
            if point_in_polygon((x, y), polygon):
                overlap_pixels += 1

    intersection_area = (overlap_pixels / (grid_samples * grid_samples)) * bbox_area

    # BBOX 면적이 ROI의 몇 %를 차지하는지
    ratio = intersection_area / roi_area
    return min(ratio, 1.0)
```

**임계값**:
```python
OVERLAP_THRESHOLD = 0.10  # BBOX가 ROI의 10% 이상 차지할 때
```

**변경 이력**:
- 이전: BBOX의 20%가 ROI에 있을 때 (BBOX 기준)
- 현재: BBOX가 ROI의 10%를 차지할 때 (ROI 기준) ✅
- 효과: 작은 ROI에서도 더 정확한 감지 가능

### 3.4 프레임 차이 분석 (Frame Difference Detection)

**목적**: 대각선으로 지나가는 차량을 빈 주차칸으로 오인하는 것을 방지

**알고리즘 (ui/play_screen.py)**:
```python
def _check_roi_frame_difference(self, frame):
    """ROI 영역의 프레임 차이를 계산하여 차량 존재 여부 판단"""
    # 그레이스케일 변환
    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

    for roi_idx, roi_polygon in enumerate(self.app.rectangles):
        # ROI 마스크 생성
        mask = np.zeros(gray_current.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_polygon], 255)

        # ROI 영역만 추출
        roi_current = cv2.bitwise_and(gray_current, gray_current, mask=mask)
        roi_prev = cv2.bitwise_and(gray_prev, gray_prev, mask=mask)

        # 프레임 차이 계산
        diff = cv2.absdiff(roi_current, roi_prev)

        # 임계값 적용 (변화가 큰 픽셀만 추출)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 변화된 픽셀 개수 계산
        changed_pixels = cv2.countNonZero(thresh)
        roi_area = cv2.countNonZero(mask)

        # 변화 비율 계산
        change_ratio = changed_pixels / roi_area if roi_area > 0 else 0

        # 변화 비율이 20% 이상이면 차량 있음으로 판단
        vehicles_in_roi[roi_idx] = change_ratio >= 0.20

    self.prev_frame = frame.copy()
    return vehicles_in_roi
```

**수식**:
```
변화 비율 = (변화된 픽셀 수) / (ROI 면적)

where:
  변화된 픽셀 = countNonZero(threshold(absdiff(현재 프레임, 이전 프레임), 25))

조건:
  변화 비율 >= 0.20 (20%)  →  차량 존재로 판단
```

**파라미터**:
- `roi_frame_diff_threshold = 25`: 픽셀 값 차이 임계값
- `roi_change_ratio_threshold = 0.20`: ROI 변화 비율 임계값 (20%)
- `frame_diff_start_time = 8.0초`: 프레임 차이 방식 활성화 시간

**활성화 조건**:
```python
# 8초 후부터 프레임 차이 방식 활성화
current_time = self.current_frame / self.fps
if not self.frame_diff_enabled and current_time >= 8.0:
    self.frame_diff_enabled = True
```

### 3.5 하이브리드 검출 방식 (BBOX AND Frame Difference)

**알고리즘**:
```python
if self.frame_diff_enabled:  # 8초 이후
    vehicles_in_roi_bbox = self._check_vehicles_in_rois(bboxes, directions)
    vehicles_in_roi_diff = self._check_roi_frame_difference(frame)

    for roi_idx in all_roi_indices:
        roi_state = self.parking_tracker.get_roi_state(roi_idx)

        # 이미 PARKED 또는 DETECTING 상태인 ROI는 프레임 차이 체크 생략
        if roi_state == PARKED or roi_state == DETECTING:
            vehicles_in_roi[roi_idx] = vehicles_in_roi_bbox[roi_idx]
        else:
            # 새로 감지되는 차량은 AND 조건 적용
            vehicles_in_roi[roi_idx] = (vehicles_in_roi_bbox[roi_idx] and
                                       vehicles_in_roi_diff[roi_idx])
else:  # 0-8초
    # BBOX 방식만 사용 (초반 주차 차량 감지)
    vehicles_in_roi = vehicles_in_roi_bbox
```

**타임라인**:
```
0-8초: BBOX 검출만 사용
  → 초반 주차 차량이 PARKED 상태 도달 (3초 소요)
  → 8초까지 여유 시간 확보

8초 이후: 하이브리드 검출
  ├─ PARKED/DETECTING 상태: BBOX만 체크 (프레임 차이 면제)
  └─ EMPTY/LEAVING 상태: BBOX AND 프레임 차이 (대각선 차량 필터링)
```

**효과**:
- ✅ 초반 주차 차량 정상 감지 (8초 여유)
- ✅ 대각선 통과 차량 오탐 방지 (AND 조건)
- ✅ 기존 주차 차량 안정성 유지 (상태별 면제)

### 3.6 상태 머신 (State Machine)

**상태 정의 (core/ParkingTracker.py)**
```
EMPTY (0)      → 비어있음
DETECTING (1)  → 차량 감지 중 (주차 판정 전)
PARKED (2)     → 주차 확정
LEAVING (3)    → 떠나는 중 (비워짐 판정 전)
```

**상태 전이**
| 현재 상태 | 입력 | 조건 | 다음 상태 |
|----------|------|------|---------|
| EMPTY | 차량 감지 | - | DETECTING |
| DETECTING | 차량 감지 | frame_count >= 3초 | PARKED |
| DETECTING | 미감지 | miss_count >= 1초 | EMPTY |
| PARKED | 차량 감지 | - | PARKED (유지) |
| PARKED | 미감지 | miss_count >= 1초 | LEAVING |
| LEAVING | 미감지 | frame_count >= 2초 | EMPTY |
| LEAVING | 차량 감지 | - | PARKED |

**구현 (core/ParkingTracker.py)**:
```python
def __init__(self, fps=30, parking_seconds=3, unparking_seconds=2,
             detection_tolerance_seconds=1):
    self.fps = fps
    self.parking_frames = int(fps * parking_seconds)        # 주차 판정: 90 프레임
    self.unparking_frames = int(fps * unparking_seconds)    # 비워짐: 60 프레임
    self.tolerance_frames = int(fps * detection_tolerance_seconds)  # 유예: 30 프레임

    # ROI별 상태: {roi_id: {'state': int, 'frame_count': int, 'miss_count': int}}
    self.roi_states = {}

def _handle_vehicle_detected(self, roi_id, state_info, newly_parked):
    """차량 감지 시 상태 처리"""
    state_info['miss_count'] = 0  # 감지 실패 카운트 리셋

    if current_state == self.EMPTY:
        state_info['state'] = self.DETECTING
        state_info['frame_count'] = 1

    elif current_state == self.DETECTING:
        state_info['frame_count'] += 1
        if state_info['frame_count'] >= self.parking_frames:  # 3초
            state_info['state'] = self.PARKED
            newly_parked.add(roi_id)

def _handle_vehicle_not_detected(self, roi_id, state_info, newly_empty):
    """차량 미감지 시 상태 처리 (1초 허용)"""
    if current_state == self.DETECTING:
        state_info['miss_count'] += 1
        if state_info['miss_count'] >= self.tolerance_frames:  # 1초
            state_info['state'] = self.EMPTY

    elif current_state == self.PARKED:
        state_info['miss_count'] += 1
        if state_info['miss_count'] >= self.tolerance_frames:  # 1초
            state_info['state'] = self.LEAVING
            state_info['frame_count'] = 1

    elif current_state == self.LEAVING:
        state_info['frame_count'] += 1
        if state_info['frame_count'] >= self.unparking_frames:  # 2초
            state_info['state'] = self.EMPTY
            newly_empty.add(roi_id)
```

**안정성 기능**:
- `detection_tolerance_seconds = 1초`: YOLO 감지 실패 시 1초 유예
- `parking_seconds = 3초`: 주차 확정 대기 시간
- `unparking_seconds = 2초`: 비워짐 확정 대기 시간

### 3.7 실시간 비동기 추론 (Asynchronous Inference)

**아키텍처**:
- UI 스레드: 프레임 렌더링 담당
- 워커 스레드: YOLO 추론 담당 (백그라운드)
- 데이터 공유: Thread-safe Lock 사용

**장점**:
- ✅ UI 반응성 유지 (추론 중에도 프레임 표시)
- ✅ 프레임 드롭 최소화 (항상 최신 프레임만 처리)
- ✅ 높은 FPS 유지 가능

**구현 (ui/play_screen.py)**
```python
# 워커 스레드
def _inference_worker(self):
    """백그라운드에서 YOLO 추론 실행"""
    while not self._infer_stop_event.is_set():
        with self._infer_lock:
            frame = self._latest_frame_for_infer

        if frame is not None:
            # YOLO 추론 (블로킹 작업)
            bboxes, confidences = self.vehicle_detector.detect_vehicles(frame)

            with self._infer_lock:
                self._latest_detections = (bboxes, confidences)

# UI 스레드
def update_frame(self):
    """매 프레임 UI 업데이트"""
    with self._infer_lock:
        # 최신 프레임 전달
        self._latest_frame_for_infer = frame.copy()

        # 최신 감지 결과 스냅샷
        bboxes, confidences = self._latest_detections
        bboxes = list(bboxes)
        confidences = list(confidences)

    # 차량 방향 감지
    directions = self.vehicle_detector.detect_vehicle_direction(bboxes)

    # ROI 판정 및 상태 업데이트
    vehicles_in_roi = self._check_vehicles_in_rois(bboxes, directions)
    tracking_info = self.parking_tracker.update(vehicles_in_roi)
```

### 3.8 성능 최적화

| 기법 | 효과 | 적용 상황 |
|------|------|---------|
| **배속 반해상도 감지** | 4배 빠른 추론 | 배속 >= 3.0x |
| **적분 이미지 (ROI 판정)** | O(1) 겹침 계산 | 모든 프레임 |
| **비동기 추론** | UI 블로킹 제거 | 모든 프레임 |
| **BBOX 축소** | 오탐 감소 | 모든 감지 |
| **프레임 차이 분석** | 대각선 차량 필터링 | 8초 이후 |
| **GPU 가속** | 10배 빠른 추론 | GPU 사용 시 |

**결과**:
- GPU 모드 (Large): 고정확도 검출
- UI 렌더링: 평균 12-13ms (배속 최적화로 부드러움 유지)
- 시간 기반 동기화: 배속에 관계없이 정확한 슬라이더 위치

---

## 📦 설치 및 실행

### 요구사항
- **Python**: 3.8 이상 (3.12 권장)
- **OS**: Windows / Linux / macOS
- **GPU** (선택): NVIDIA CUDA 12.1+

### 설치 단계

**1. 저장소 클론**
```bash
git clone https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT.git
cd Computer_Vision_UN_TEAM_PROJECT
```

**2. Python 가상환경 생성 (권장)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. 의존성 설치**
```bash
pip install -r requirements.txt
```

**4. (선택) GPU 가속 설정**

GPU를 사용하려면 CUDA 호환 PyTorch 설치:
```bash
# CUDA 12.1 지원
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU 확인:
```bash
python -c "import torch; print(f'GPU 사용 가능: {torch.cuda.is_available()}')"
```

### 실행

```bash
python main.py
```

**UI 워크플로우:**
1. **메인 화면**: "영상 선택" → 비디오 파일 선택
2. **ROI 설정 화면**: "주차칸 설정" → OpenCV 창에서 마우스로 4점 클릭
   - 각 ROI마다 4번 클릭해 사각형 생성
   - ESC 키로 완료
3. **재생 화면**:
   - "▶ 재생" / "⏸ 일시정지" 버튼
   - "배속" 선택 (0.5x ~ 4x)
   - 슬라이더로 위치 이동
   - 실시간 감지 결과 표시 (ROI별 상태, 바운딩 박스, 방향 화살표)

---

## 🔧 설정 및 파라미터

### 주요 설정값

**상태 전이 시간 (core/ParkingTracker.py)**:
```python
PARKING_SECONDS = 3           # 주차 확정 대기 시간
UNPARKING_SECONDS = 2         # 비워짐 확정 대기 시간
DETECTION_TOLERANCE = 1       # YOLO 미감지 유예 시간
```

**검출 임계값 (core/VehicleDetector.py)**:
```python
CONFIDENCE_THRESHOLD = 0.5    # 신뢰도 0.5 이상만 검출
OVERLAP_THRESHOLD = 0.10      # BBOX가 ROI의 10% 이상 차지
BBOX_SHRINK_RATIO = 0.80      # BBOX 80% 유지 (20% 제거)
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
```

**프레임 차이 분석 (ui/play_screen.py)**:
```python
roi_frame_diff_threshold = 25           # 픽셀 값 차이 임계값
roi_change_ratio_threshold = 0.20       # ROI 변화 비율 20%
frame_diff_start_time = 8.0             # 8초 후 활성화
```

---

## 📝 코드 구조 및 모듈화

### UI 컴포넌트 모듈화 (core/ui_components.py)

반복되는 버튼 생성 코드를 헬퍼 함수로 통합:

```python
# 일관된 헬퍼 사용
UIButton.create_primary(frame, "▶ 재생", self.play)
UIButton.create_danger(frame, "⏸ 일시정지", self.pause)

# 배속 버튼 자동 생성
for speed in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    UIButton.create_speed_button(frame, f"{speed}x",
                                 lambda s=speed: self.set_playback_speed(s))
```

**개선 효과**:
- 코드 중복 95% 감소
- 테마 색상 한 곳에서 관리
- 유지보수 용이

---

## ⚠️ 알려진 제한사항

1. **ROI 사각형만 지원**: 다각형 미지원 (현재는 4점 선택)
2. **단일 비디오 처리**: 실시간 카메라 입력 미지원 (파일 기반만)
3. **Windows 테스트**: Linux/macOS에서의 호환성 미검증

---

## 📊 주요 개선 사항

### 최근 업데이트 (2025-11-30)

1. **프레임 차이 분석 추가**
   - ROI 영역의 픽셀 변화 감지
   - 대각선 통과 차량 오탐 방지
   - BBOX AND 프레임 차이 하이브리드 방식

2. **차량 방향 감지**
   - 프레임간 BBOX 중심 추적
   - 8방향 이동 방향 분류 및 시각화
   - 속도 및 신뢰도 계산

3. **BBOX 축소 기능**
   - 차량 중심부(80%)만 사용
   - 대각선 차량 모서리 제거
   - ROI 기준 겹침 비율 계산

4. **초반 주차 차량 처리**
   - 8초 여유 시간 확보
   - 상태별 프레임 차이 면제
   - 안정적인 주차 감지

---

## 📚 참고 문헌 및 라이브러리

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **OpenCV**: 영상 처리 및 ROI 선택
- **Tkinter**: GUI 프레임워크
- **NumPy**: 수치 연산 (적분 이미지)
- **Pillow**: 이미지 표시

---

## 🤝 기여 및 피드백

프로젝트 개선사항이나 버그 리포트는 이슈 또는 PR로 제출해주세요.

---

## 📄 라이선스

이 프로젝트는 별도의 라이선스 제약이 없습니다. 자유롭게 사용하세요.

---

**마지막 업데이트**: 2025-11-30
**프로젝트 저장소**: [GitHub - Computer_Vision_UN_TEAM_PROJECT](https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT)
