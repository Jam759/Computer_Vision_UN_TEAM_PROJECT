# 🅿️ Parking Monitor System

> YOLOv8 기반 실시간 주차 감지 및 모니터링 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 구조](#시스템-구조)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설정 옵션](#설정-옵션)
- [문제 해결](#문제-해결)

---

## 개요

**Parking Monitor System**은 YOLOv8을 이용한 실시간 주차 감지 시스템입니다. 
사용자가 영상에서 ROI(관심 영역)를 선택하면, 시스템이 자동으로 차량 진입/이탈을 감지하고 주차 상태를 추적합니다.

### 핵심 특징
- 🎯 **실시간 차량 감지**: YOLOv8 Medium 모델 기반
- 📊 **상태 머신 기반 추적**: 4가지 상태(EMPTY, DETECTING, PARKED, LEAVING)
- 🛡️ **안정적인 감지**: YOLO 실패 시 1초 허용 기간
- 🎨 **직관적인 UI**: Tkinter 기반 사용자 인터페이스
- ✔️ **ROI 유효성 검증**: 다각형 영역 검증

---

## 주요 기능

### 1. 영상 선택 (Main Screen)
- 비디오 파일 선택 (.mp4, .avi, .mov, .mkv)
- 썸네일 미리보기
- 지원 형식: MP4, AVI, MOV, MKV

### 2. ROI 설정 (ROI Choice Screen)
- 마우스 클릭으로 주차칸 선택 (4점 다각형)
- 유효성 검증 (4개 점, 최소 면적 100px²)
- 실시간 미리보기

### 3. 실시간 모니터링 (Play Screen)
- 주차 상태 시각화
- 프레임별 차량 감지
- 주차 통계 표시
  - 전체 주차칸 수
  - 주차된 차량 수
  - 빈 주차칸 수

---

## 시스템 구조

### 상태 머신 (State Machine)

```
EMPTY (비어있음)
  ↓ [차량 감지]
DETECTING (감지 중, 3초)
  ↓ [확정]
PARKED (주차됨)
  ↓ [미감지 1초]
LEAVING (떠나는 중, 5초)
  ↓ [확정]
EMPTY (비워짐)
```

### 감지 로직

```
YOLO 감지 → ROI 겹침 확인 → 상태 업데이트
           ↓
         30% 이상 겹침?
         (조정 가능: OVERLAP_THRESHOLD)
```

---

## 설치 및 실행

### 사전 요구사항
- Python 3.8 이상
- CUDA (GPU 사용 시, 선택사항)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT.git
cd computerVisionProject

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install opencv-python tkinter pillow ultralytics numpy
```

### 모델 다운로드

프로젝트 루트 디렉토리에 YOLO 모델 파일이 필요합니다:

```bash
# YOLOv8 Medium 모델 (권장)
# 파일: yolov8m.pt (약 50MB)
# 자동으로 다운로드됩니다 (첫 실행 시)
```

### 실행

```bash
python main/main.py
```

---

## 사용 방법

### 단계 1: 영상 선택
1. "영상 선택" 버튼 클릭
2. 비디오 파일 선택
3. 썸네일 확인 후 "주차칸 선택 화면으로 이동" 클릭

### 단계 2: ROI 설정
1. "주차칸 설정(OpenCV 창)" 버튼 클릭
2. 각 주차칸의 4개 모서리를 마우스로 클릭
3. 사각형이 그려지면 다음 주차칸 선택
4. ESC 키로 종료

### 단계 3: 실시간 모니터링
1. "영상 재생 화면 →" 클릭
2. 재생/일시정지 버튼으로 제어
3. 슬라이더로 프레임 이동
4. 상단 패널에서 실시간 통계 확인

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| **영상 처리** | OpenCV 4.5+ |
| **차량 감지** | YOLOv8 Medium |
| **UI** | Tkinter |
| **이미지 처리** | Pillow |
| **수치 계산** | NumPy |
| **모델** | Ultralytics YOLO |

---

## 프로젝트 구조

```
computerVisionProject/
├── main/
│   └── main.py                 # 애플리케이션 진입점
├── app/
│   └── parking_app.py          # 메인 앱 컨테이너
├── ui/
│   ├── main_screen.py          # 영상 선택 화면
│   ├── roi_choice_screen.py    # ROI 설정 화면
│   └── play_screen.py          # 실시간 모니터링 화면
├── core/
│   ├── VehicleDetector.py      # YOLO 기반 차량 감지
│   ├── ParkingTracker.py       # 상태 머신 기반 추적
│   ├── rectangle_selector.py   # 마우스 ROI 선택
│   └── utils.py                # 공통 유틸 함수
├── yolov8m.pt                  # YOLO 모델 (자동 다운로드)
├── yolov8n.pt                  # YOLO 나노 모델 (선택사항)
├── requirements.txt            # 의존성
└── README.md                   # 이 파일
```

---

## 설정 옵션

### PlayScreen 설정 (`ui/play_screen.py`)

```python
# 주차 확정 시간 (초)
PARKING_SECONDS = 3

# 비워짐 확정 시간 (초)
UNPARKING_SECONDS = 5

# 프레임 업데이트 주기 (ms)
UPDATE_INTERVAL = 33  # ≈ 30fps
```

### VehicleDetector 설정 (`core/VehicleDetector.py`)

```python
# YOLO 신뢰도 임계값 (0.0 ~ 1.0)
CONFIDENCE_THRESHOLD = 0.5

# ROI 겹침 비율 임계값 (0.0 ~ 1.0)
# 30% 이상 겹침이 필요
OVERLAP_THRESHOLD = 0.3  # 또는 0.2, 0.4 등으로 조정
```

### ParkingTracker 설정 (`core/ParkingTracker.py`)

```python
# YOLO 감지 실패 허용 시간 (초)
detection_tolerance_seconds = 1
```

---

## 차량 클래스

COCO 데이터셋 기준 감지 대상:

| 클래스 ID | 차량 타입 |
|----------|---------|
| 2 | 자동차 (car) |
| 3 | 오토바이 (motorcycle) |
| 5 | 버스 (bus) |
| 7 | 트럭 (truck) |

---

## 색상 정의

상태별 ROI 표시 색상 (BGR 형식):

| 상태 | 색상 | RGB |
|------|------|-----|
| EMPTY | 초록색 | (0, 255, 0) |
| DETECTING | 주황색 | (0, 165, 255) |
| PARKED | 빨강색 | (0, 0, 255) |
| LEAVING | 밝은 초록 | (0, 255, 165) |

---

## 모델 선택

### YOLOv8 Medium (권장) ✅
- 파일: `yolov8m.pt` (~50MB)
- 정확도: ⭐⭐⭐⭐⭐
- 속도: ⭐⭐⭐
- 추천: 정확도 중시

### YOLOv8 Nano (빠름)
- 파일: `yolov8n.pt` (~10MB)
- 정확도: ⭐⭐⭐
- 속도: ⭐⭐⭐⭐⭐
- 추천: 속도 중시

모델 변경:
```python
# VehicleDetector.py에서
self.model = YOLO('yolov8n.pt')  # nano로 변경
```

---

## 문제 해결

### 1. 차량이 감지되지 않음

**확인사항:**
```
□ 영상 파일이 손상되지 않았는지 확인
□ YOLO 모델이 다운로드되었는지 확인 (yolov8m.pt 존재)
□ ROI가 올바르게 설정되었는지 확인
□ 신뢰도 임계값 조정: CONFIDENCE_THRESHOLD 0.5 → 0.3
□ 겹침 임계값 조정: OVERLAP_THRESHOLD 0.3 → 0.2
```

### 2. 프로그램이 느림

**해결 방법:**
```
□ YOLOv8 Nano 모델로 변경: yolov8n.pt
□ 영상 해상도 낮추기
□ GPU 사용 확인 (CUDA 설치)
□ 불필요한 배경 프로세스 종료
```

### 3. ROI 설정 오류

**확인사항:**
```
□ 정확히 4개 점을 클릭했는지 확인
□ 점들이 유효한 다각형을 형성하는지 확인
□ 다각형 면적이 충분한지 확인 (최소 100px²)
```

### 4. 주차 상태 변화가 너무 빠름

**조정 방법:**
```python
# play_screen.py에서
PARKING_SECONDS = 3      # → 5로 증가
UNPARKING_SECONDS = 5    # → 8로 증가
```

### 5. YOLO 모델 다운로드 오류

```bash
# 수동으로 다운로드
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

---

## 성능 최적화 팁

1. **영상 전처리**
   - 고해상도 영상은 다운샘플링 권장
   - FPS 확인: 30fps 이상 유지

2. **ROI 설정**
   - 주차칸 크기에 맞게 정확히 선택
   - 너무 작거나 크지 않도록 주의

3. **감지 파라미터**
   - 조명 조건에 맞게 임계값 조정
   - 테스트 영상으로 최적값 찾기

4. **시스템 리소스**
   - GPU 사용 권장 (Nvidia CUDA)
   - RAM 4GB 이상 권장

---

## 로그 및 디버깅

콘솔 출력 확인:

```
[INFO] 영상 로드 성공
[INFO] 프레임: 100/500
[DEBUG] ROI 1: PARKED
[ERROR] 프레임 읽기 실패
```

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

## 기여 가이드

```bash
# 1. Fork 또는 Clone
git clone https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT.git

# 2. Feature 브랜치 생성
git checkout -b feature/새기능

# 3. 변경 사항 커밋
git commit -m "feat: 새로운 기능 추가"

# 4. Pull Request 생성
git push origin feature/새기능
```

---

## 주요 개선 사항

### v2.0 (현재)
- ✅ 상태 머신 기반 추적
- ✅ YOLO 감지 실패 허용 (1초)
- ✅ ROI 유효성 검증
- ✅ 전역 에러 처리
- ✅ 리소스 정리 로직

### v1.0
- 기본 YOLO 감지
- 영상 재생 UI
- ROI 선택 기능

---

## 연락처 및 문의

- **GitHub**: [Jam759](https://github.com/Jam759)
- **Repository**: [Computer_Vision_UN_TEAM_PROJECT](https://github.com/Jam759/Computer_Vision_UN_TEAM_PROJECT)

---

## 감사의 말

- YOLOv8 모델: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV: [OpenCV Team](https://opencv.org)
- COCO Dataset: [Linköping University](https://cocodataset.org)

---

## 참고 자료

- [YOLOv8 공식 문서](https://docs.ultralytics.com)
- [OpenCV 튜토리얼](https://docs.opencv.org/master)
- [COCO Dataset](https://cocodataset.org)

---

**마지막 업데이트**: 2025년 11월 25일  
**버전**: 2.0.0  
**상태**: ✅ Production Ready
