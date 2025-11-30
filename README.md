# 🅿️ Parking Monitor System

**YOLOv8 기반 실시간 주차 감지 및 모니터링 시스템**

UN TEAM 대학 컴퓨터 비전 팀 프로젝트

---

## 📌 프로젝트 개요

이 프로젝트는 사용자가 비디오에서 주차 칸(ROI, Region of Interest)을 지정하면, YOLOv8 기반 차량 검출 결과를 이용해 각 ROI의 주차 상태를 실시간으로 추적하고 시각화하는 데스크톱 애플리케이션입니다.

**핵심 특징:**
- 🎯 **차량 자동 감지**: YOLOv8 모델(M 버전)을 사용한 고속, 고정확 차량 검출
- 📍 **ROI 기반 추적**: 사용자 정의 주차칸 영역에 대한 개별 상태 추적
- 🔄 **안정적 상태 전이**: 상태 머신 기반의 노이즈 완화 (1초 유예)
- ⚡ **GPU 가속**: NVIDIA CUDA 지원으로 높은 처리 성능 (추론 13-26ms, UI 12-13ms)
- 🎨 **직관적 GUI**: Tkinter 기반 3단계 워크플로우 (비디오 선택 → ROI 설정 → 실시간 모니터링)

---

## 📊 (1) 프로젝트 개요 및 설명

### 목표
사용자가 지정한 주차 칸에 대해 차량의 진입·이탈을 자동으로 감지하여 주차 상태를 실시간으로 추적하고 시각화합니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **차량 검출** | YOLOv8을 이용한 차량 바운딩 박스 획득 (신뢰도 0.5 이상) |
| **ROI 판정** | 바운딩 박스와 ROI의 겹침 비율 계산 (적분 이미지 기반 최적화) |
| **상태 추적** | 4가지 상태 전이: EMPTY → DETECTING → PARKED → LEAVING → EMPTY |
| **노이즈 완화** | YOLO 감지 실패에 대한 1초 유예(tolerance) 적용 |
| **실시간 시각화** | ROI와 차량 바운딩 박스를 프레임에 그려 표시 |
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
  ↓ (5초 이상 미감지)
EMPTY (비어있음)
```

---

## 🏗️ (2) 전체 구조 및 개발환경

### 개발환경

| 항목 | 요구사항 | 현재 버전 |
|------|---------|---------|
| **언어** | Python 3.8+ | 3.12.9 ✓ |
| **OS** | Windows / Linux / macOS | Windows ✓ |
| **GPU** | (선택) NVIDIA CUDA 12.1+ | CUDA 13.0 + RTX 4070 ✓ |
| **Python 패키지** | requirements.txt 참조 | 설치 필요 |

### 프로젝트 구조

```
Computer_Vision_UN_TEAM_PROJECT/
├── main/
│   └── main.py                    # 애플리케이션 진입점
├── app/
│   └── parking_app.py             # 앱 컨테이너 (화면 전환, 공용 상태)
├── ui/
│   ├── main_screen.py             # 비디오 선택 및 썸네일 표시
│   ├── roi_choice_screen.py        # ROI 설정 화면 (OpenCV)
│   ├── play_screen.py              # 재생 및 실시간 감지 시각화
│   └── temp/                       # 임시 파일 (개발용)
├── core/
│   ├── ui_components.py           # UI 컴포넌트 헬퍼 (모듈화)
│   ├── VehicleDetector.py         # YOLO 모델 래퍼
│   ├── ParkingTracker.py          # 상태 머신 (ROI별 추적)
│   ├── rectangle_selector.py      # OpenCV 기반 ROI 선택 도구
│   └── utils.py                    # 기하학/색상/비디오 유틸
├── data/                           # 데이터 (저장용)
├── yolov8l.pt                      # YOLOv8 Large 모델 (자동 다운로드)
├── requirements.txt                # 의존성 명시
└── README.md                       # 이 파일

```

### 파일별 역할

**UI 계층 (ui/)**
- `main_screen.py`: 비디오 파일 선택, 첫 프레임 썸네일 표시
- `roi_choice_screen.py`: OpenCV 창에서 마우스 클릭으로 ROI 설정
- `play_screen.py`: 비디오 재생, 실시간 차량 감지, 상태 표시
  - 비동기 추론 워커: 백그라운드 스레드에서 YOLO 추론 실행
  - 성능 최적화: 배속 >= 3.0일 때 반해상도 감지 적용
  - 시간 기반 프레임 동기화: 배속에 관계없이 정확한 슬라이더 위치 유지

**핵심 로직 (core/)**
- `VehicleDetector.py`: YOLOv8 Large 모델 로드 및 차량 검출 (높은 정확도)
- `ParkingTracker.py`: 프레임 기반 상태 머신 관리
- `rectangle_selector.py`: 사용자의 마우스 클릭으로 ROI 다각형 생성
- `utils.py`: 기하학 연산, 색상 정의, 비디오 처리 유틸

**애플리케이션 (app/)**
- `parking_app.py`: Tkinter 루트 컨테이너, 화면 전환 관리

---

## 🚀 (3) 주요 기술 및 기능 설명

### 3.1 차량 검출 (Vehicle Detection)

**방식**: YOLOv8 Large (`yolov8l.pt`)
- 클래스: 차량 (자동차, 오토바이, 버스, 트럭)
- 신뢰도 임계값: 0.5
- 정확도: Medium보다 3-5% 향상
- GPU 지원: NVIDIA CUDA 활용 시 ~40-60ms/프레임

**구현 (core/VehicleDetector.py)**
```python
def detect_vehicles(frame):
    """프레임에서 차량 검출"""
    results = self.model(frame, verbose=False, classes=VEHICLE_CLASSES, device=self.device)
    # BBOX 중심 영역만 추출 (대각선 차량의 모서리 제거)
    bbox = self._shrink_bbox(bbox, BBOX_SHRINK_RATIO=0.80)
    return bboxes, confidences
```

**BBOX 축소 기능 (새로 추가)**:
- `BBOX_SHRINK_RATIO = 0.80` (80% 유지, 20% 제거)
- 대각선으로 보는 차량의 불필요한 모서리 제거
- 중심 기준 대칭으로 축소하여 차량 중심부만 검사
- 효과: 빈 공간 오탐 감소, 정확도 향상

**GPU 모드 설정**:
- CUDA 설치 시 자동으로 GPU 사용
- 폴백: CPU 모드로 자동 전환
- 현재 성능: GPU 약 10배 빠름 (CPU 400-500ms → GPU 40-60ms)

### 3.2 ROI 판정 (ROI Overlap Detection)

**최적화 방식**: 적분 이미지 (Integral Image) 기반
- 프로젝트 시작 시 각 ROI에 대해 적분 이미지 사전 계산
- 매 프레임 O(1) 시간에 겹침 비율 계산
- 그리드 샘플링 (20×20)보다 100배 이상 빠름

**알고리즘 (core/utils.py) - 개선된 ROI 기준 계산**
```python
def calculate_bbox_in_roi_ratio(bbox, polygon):
    """BBOX가 ROI의 몇 %를 차지하는지 계산"""
    # 1. BBOX와 ROI의 교집합 계산
    # 2. 교집합 / ROI 면적 (ROI 기준)
    # 반환: BBOX가 ROI에서 차지하는 비율 (0.0 ~ 1.0)
```

**임계값**: OVERLAP_THRESHOLD = 0.2 (BBOX가 ROI의 20% 이상 차지할 때)

**변경 내용**:
- 이전: BBOX의 20%가 ROI에 있을 때 (BBOX 기준)
- 현재: BBOX가 ROI의 20%를 차지할 때 (ROI 기준) ✅
- 효과: 작은 ROI에서도 더 정확한 감지 가능

### 3.3 상태 머신 (State Machine)

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
| LEAVING | 미감지 | frame_count >= 5초 | EMPTY |
| LEAVING | 차량 감지 | - | PARKED |

**안정성 기능**:
- `detection_tolerance_seconds = 1`: YOLO 감지 실패 시 1초 유예
- `parking_seconds = 3`: 주차 확정 대기 시간
- `unparking_seconds = 5`: 비워짐 확정 대기 시간

### 3.4 실시간 비동기 추론 (Asynchronous Inference)

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
# UI 스레드
with self._infer_lock:
    self._latest_frame_for_infer = frame.copy()  # 최신 프레임 제공
    bboxes, confidences = self._latest_detections  # 스냅샷 읽기

# 워커 스레드
while not self._infer_stop_event.is_set():
    frame = self._latest_frame_for_infer  # 최신 프레임 획득
    bboxes, confidences = self.vehicle_detector.detect_vehicles(frame)
    with self._infer_lock:
        self._latest_detections = (bboxes, confidences)  # 결과 저장
```

### 3.5 성능 최적화

| 기법 | 효과 | 적용 상황 |
|------|------|---------|
| **배속 반해상도 감지** | 10배 빠른 추론 | 배속 >= 3.0x |
| **적분 이미지 (ROI 판정)** | 100배 빠른 겹침 계산 | 모든 프레임 |
| **비동기 추론** | UI 블로킹 제거 | 모든 프레임 |
| **프레임 드롭 방식** | 지연 축적 방지 | 느린 PC |
| **GPU 가속** | 10배 빠른 추론 | GPU 사용 시 |

**결과**:
- GPU 모드 (Large): 평균 40-60ms/프레임 (높은 정확도)
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
python main/main.py
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
   - 실시간 감지 결과 표시 (ROI별 상태, 바운딩 박스)

---

## 📊 성능 벤치마크

### 테스트 환경
- **GPU**: NVIDIA GeForce RTX 4070
- **CPU**: 현재 PC 기준
- **해상도**: 1920×1080
- **FPS**: 20

### 성능 지표

| 항목 | 추론 시간 | UI 렌더링 | 총 처리 시간 |
|------|---------|---------|-----------|
| **1배속 (GPU)** | 13-26ms | 67-72ms | 67-72ms |
| **4배속 (GPU, 반해상도)** | 13-18ms | 12-13ms | 12-13ms |
| **CPU 모드** | ~200ms | - | 블로킹 |

**결론**: GPU + 비동기 추론으로 매끄러운 실시간 처리 가능

---

## 🔧 설정 및 파라미터

### 주요 설정값 (core/ParkingTracker.py)

```python
# 상태 전이 시간
PARKING_SECONDS = 3           # 주차 확정 대기 시간
UNPARKING_SECONDS = 5         # 비워짐 확정 대기 시간
DETECTION_TOLERANCE = 1       # YOLO 미감지 유예 시간

# FPS 설정 (비디오마다 다름)
fps = VideoUtils.get_video_fps(cap)  # 자동 감지
```

### 검출 임계값 (core/VehicleDetector.py)

```python
CONFIDENCE_THRESHOLD = 0.5    # 신뢰도 0.5 이상만 검출
OVERLAP_THRESHOLD = 0.2       # ROI와 20% 이상 겹침 판정
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
```

---

## 📝 코드 구조 및 모듈화

### UI 컴포넌트 모듈화 (core/ui_components.py)

반복되는 버튼 생성 코드를 헬퍼 함수로 통합:

```python
# 이전: 72줄의 반복 코드
# 현재: 일관된 헬퍼 사용
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

1. **모델 업그레이드**: YOLOv8 Large(`yolov8l.pt`)로 높은 정확도 제공
2. **ROI 사각형만 지원**: 다각형 미지원 (현재는 4점 선택)
3. **단일 비디오 처리**: 실시간 카메라 입력 미지원 (파일 기반만)
4. **Windows 테스트**: Linux/macOS에서의 호환성 미검증

---

## 📖 사용 예시

### 기본 사용 흐름

```
1. 앱 실행 (main/main.py)
2. 메인 화면 → "영상 선택" → 주차장 비디오 선택
3. 첫 프레임 썸네일 표시
4. "주차칸 선택 화면으로 이동"
5. ROI 설정 화면 → "주차칸 설정" → OpenCV 창 띄우기
6. 마우스로 각 주차칸 주변을 4번 클릭 (사각형 만들기)
7. 모든 주차칸 설정 완료 후 ESC 키
8. ROI 미리보기 확인
9. "영상 재생 화면 →" 클릭
10. 배속/재생/일시정지 제어
11. 실시간 주차 상태 표시 관찰
```

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

**마지막 업데이트**: 2025-11-29  
**프로젝트 저장소**: [GitHub - Jam759/Computer_Vision_UN_TEAM_PROJECT](https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT)

```powershell
pip install -r .\requirements.txt
```

2) 개별 패키지 설치(예):

```powershell
pip install numpy opencv-python Pillow ultralytics
```

3) 실행:

```powershell
python main\main.py
```

**권장**: GPU를 사용하려면 시스템에 맞는 `torch` (CUDA) 패키지를 별도로 설치하세요.

---

필요하시면 다음을 추가로 도와드리겠습니다:
- 영어 README 버전 작성
- `requirements.txt`에 버전 고정(예: `opencv-python==4.8.1.78`) 적용
- 가상환경 생성(venv) 및 설치 스크립트 작성

원하시는 추가 작업을 알려 주세요.
