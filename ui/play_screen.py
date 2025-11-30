"""
영상 재생 및 실시간 주차 감지 화면
"""

import tkinter as tk
import cv2
import time
import numpy as np
import threading
from PIL import Image, ImageTk
from core.VehicleDetector import VehicleDetector
from core.ParkingTracker import ParkingTracker
from core.utils import VideoUtils, ColorUtils, GeometryUtils
from core.ui_components import UIStyle, UIButton, UILabel, StatisticsFormatter


class PlayScreen:
    """
    비디오 재생 및 실시간 차량 감지, 주차 상태 시각화
    """
    
    # 재생 설정 (안정성 향상)
    PARKING_SECONDS = 3      # 주차 인정 시간 (초) - 안정성 높음
    UNPARKING_SECONDS = 5    # 비워짐 인정 시간 (초) - 안정성 높음
    # UPDATE_INTERVAL은 실제 비디오 FPS에 따라 동적으로 계산됨 (__init__에서 설정)
    
    def __init__(self, app):
        """
        Args:
            app: ParkingApp 인스턴스 (영상 경로, ROI 정보 포함)
        """
        self.app = app
        
        try:
            # 비디오 로드
            self.cap = cv2.VideoCapture(app.video_path)
            if not self.cap.isOpened():
                raise Exception("영상을 열 수 없습니다")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames <= 0:
                raise Exception("영상 프레임 정보를 읽을 수 없습니다")
            
            self.current_frame = 0
            self.is_playing = False  # 초기 상태: 일시정지
            self.frame_loaded = False  # 현재 프레임이 로드되었는지 여부
            self.frame_times = []  # 프레임 처리 시간 측정용
            self.frames_to_skip = 0  # 밀린 프레임 개수
            self.playback_speed = 1.0  # 재생 배속 (1.0 = 정상, 2.0 = 2배속, 0.5 = 0.5배속)
            self.playback_time = 0.0  # 누적 재생 시간 (초 단위, 배속 반영)
            
            # FPS 정보
            self.fps = VideoUtils.get_video_fps(self.cap)
            # 실제 FPS에 따라 UPDATE_INTERVAL 계산 (밀리초 단위)
            self.update_interval = max(1, int(1000.0 / self.fps))  # 최소 1ms
            
            # 차량 감지기 및 주차 추적기
            self.vehicle_detector = VehicleDetector()
            self.parking_tracker = ParkingTracker(
                fps=self.fps,
                parking_seconds=self.PARKING_SECONDS,
                unparking_seconds=self.UNPARKING_SECONDS,
                detection_tolerance_seconds=1  # YOLO 감지 실패 1초 허용
            )

            # ROI별 마스크 적분 이미지(precomputed) 준비 -> 빠른 겹침 검사
            self._prepare_roi_integrals()

            # 프레임 차이 기반 주차 감지를 위한 이전 프레임 저장
            self.prev_frame = None
            self.roi_frame_diff_threshold = 25  # 프레임 차이 임계값 (픽셀값 차이)
            self.roi_change_ratio_threshold = 0.20  # ROI 영역 변화 비율 임계값 (20% 이상 변화)
            self.frame_diff_start_time = 8.0  # 프레임 차이 방식 활성화 시간 (초) - 초반 주차 차량이 PARKED 도달할 시간 확보
            self.frame_diff_enabled = False  # 프레임 차이 방식 활성화 플래그

            # 비동기 추론 관련 초기화
            self._latest_frame_for_infer = None
            self._latest_detections = ([], [])  # (bboxes, confidences)
            self._infer_lock = threading.Lock()
            self._infer_stop_event = threading.Event()
            # Start inference worker thread (daemon)
            self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self._infer_thread.start()
            
            # UI 구성
            self._build_ui()
            self.update_frame()
        
        except Exception as e:
            self.cap = None
            self.vehicle_detector = None
            self.parking_tracker = None
            self._build_error_ui(str(e))
    
    def _build_ui(self):
        """UI 컴포넌트 구성"""
        # 메인 프레임
        self.frame = tk.Frame(self.app.container, bg=self.app.bg_primary)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 헤더 (네비게이션 + 제목)
        self._build_header()
        
        # 정보 패널 (전체/주차됨/빈칸)
        self._build_info_panel()
        
        # 영상 표시 영역
        self._build_video_panel()
        
        # 재생 제어 바
        self._build_control_bar()
    
    def _build_header(self):
        """헤더 영역 구성"""
        header_frame = tk.Frame(self.frame, bg=UIStyle.BG_PRIMARY)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 네비게이션 버튼
        nav_frame = tk.Frame(header_frame, bg=UIStyle.BG_PRIMARY)
        nav_frame.pack(side=tk.LEFT)
        
        UIButton.create_primary(nav_frame, "← 주차칸 다시 설정", self.app.show_roi, padx=6, pady=6).pack(side=tk.LEFT, padx=3)
        UIButton.create_danger(nav_frame, "← 메인", self.app.show_main, padx=6, pady=6).pack(side=tk.LEFT, padx=3)
        
        # 제목
        UILabel.create_title(header_frame, "영상 재생 화면", UIStyle.BG_PRIMARY).pack(side=tk.LEFT, expand=True)
    
    def _build_info_panel(self):
        """정보 패널 (전체/주차됨/빈칸) 구성"""
        info = tk.Frame(self.frame, bg=UIStyle.BG_SECONDARY,
                       highlightbackground=UIStyle.ACCENT_CYAN, highlightthickness=1)
        info.pack(pady=5)
        
        total = len(self.app.rectangles)
        
        self.total_label = tk.Label(info, text=f"전체: {total}",
                                   fg=UIStyle.TEXT_PRIMARY, bg=UIStyle.BG_SECONDARY,
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.total_label.pack(side=tk.LEFT, padx=5)
        
        self.parked_label = tk.Label(info, text="주차됨: 0",
                                    fg=UIStyle.ACCENT_RED, bg=UIStyle.BG_SECONDARY,
                                    font=("Arial", 11, "bold"), padx=15, pady=8)
        self.parked_label.pack(side=tk.LEFT, padx=5)
        
        self.empty_label = tk.Label(info, text=f"빈칸: {total}",
                                   fg=UIStyle.ACCENT_CYAN, bg=UIStyle.BG_SECONDARY,
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.empty_label.pack(side=tk.LEFT, padx=5)
    
    def _build_video_panel(self):
        """영상 표시 영역 구성"""
        video_frame = tk.Frame(self.frame, bg=UIStyle.BG_SECONDARY,
                              highlightbackground=UIStyle.ACCENT_CYAN, highlightthickness=2)
        video_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.video_panel = tk.Label(video_frame, bg=UIStyle.BG_SECONDARY)
        self.video_panel.pack(fill=tk.BOTH, expand=True)
    
    def _build_error_ui(self, error_message):
        """에러 UI 구성"""
        self.frame = tk.Frame(self.app.container, bg=UIStyle.BG_PRIMARY)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 에러 메시지
        tk.Label(self.frame, text="❌ 오류 발생", 
                font=("Arial", 24, "bold"),
                fg=UIStyle.ACCENT_RED, bg=UIStyle.BG_PRIMARY).pack(pady=20)
        
        tk.Label(self.frame, text=error_message,
                font=("Arial", 12), fg=UIStyle.TEXT_SECONDARY,
                bg=UIStyle.BG_PRIMARY, wraplength=400).pack(pady=10)
        
        # 돌아가기 버튼
        UIButton.create_danger(self.frame, "← 메인으로 돌아가기", self._cleanup_and_go_main, padx=20, pady=10).pack(pady=20)
    
    def _build_control_bar(self):
        """재생 제어 바 구성"""
        control_frame = tk.Frame(self.frame, bg=UIStyle.BG_PRIMARY)
        control_frame.pack(pady=5, fill=tk.X, padx=10)
        
        # 재생 버튼
        UIButton.create_primary(control_frame, "▶ 재생", self.play, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # 일시정지 버튼
        UIButton.create_danger(control_frame, "⏸ 일시정지", self.pause, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # 배속 조절 버튼
        speed_frame = tk.Frame(control_frame, bg=UIStyle.BG_PRIMARY)
        speed_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(speed_frame, text="배속:", fg=UIStyle.TEXT_PRIMARY,
                bg=UIStyle.BG_PRIMARY, font=UIStyle.FONT_TINY).pack(side=tk.LEFT, padx=2)
        
        # 배속 버튼들 (반복 코드 제거)
        speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for speed in speeds:
            is_active = (speed == 1.0)
            btn = UIButton.create_speed_button(speed_frame, f"{speed}x", lambda s=speed: self.set_playback_speed(s), is_active)
            btn.pack(side=tk.LEFT, padx=2)
        
        self.speed_label = tk.Label(speed_frame, text="1.0x",
                                   fg=UIStyle.TEXT_PRIMARY, bg=UIStyle.BG_PRIMARY,
                                   font=("Arial", 9, "bold"), padx=5)
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # 슬라이더
        slider_frame = tk.Frame(control_frame, bg=UIStyle.BG_PRIMARY)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(slider_frame, text="위치:", fg=UIStyle.TEXT_PRIMARY,
                bg=UIStyle.BG_PRIMARY, font=UIStyle.FONT_LABEL).pack(side=tk.LEFT)
        
        self.slider = tk.Scale(slider_frame, from_=0, to=self.total_frames-1,
                              orient=tk.HORIZONTAL,
                              bg=UIStyle.BG_SECONDARY, fg=UIStyle.TEXT_PRIMARY,
                              highlightbackground=UIStyle.ACCENT_CYAN,
                              length=200)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        # 나중에 바인드해서 수동 드래그만 감지하도록 설정
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        
        self.time_label = tk.Label(slider_frame, text="00:00:00:00 / 00:00:00:00",
                                   fg=UIStyle.TEXT_SECONDARY, bg=UIStyle.BG_PRIMARY,
                                   font=UIStyle.FONT_TINY)
        self.time_label.pack(side=tk.LEFT, padx=5)
    
    def update_frame(self):
        """각 프레임 처리 및 업데이트"""
        frame_start_time = time.time()
        
        try:
            # 프레임 읽기
            if self.is_playing:
                # 재생 중: 누적 시간 기반으로 프레임 계산
                frame_interval = 1.0 / self.fps  # 한 프레임의 시간 (초)
                self.playback_time += frame_interval * self.playback_speed  # 배속 반영
                self.current_frame = int(self.playback_time * self.fps)  # 프레임 번호로 변환
                
                # 범위 체크
                if self.current_frame >= self.total_frames:
                    self.current_frame = self.total_frames - 1
                    self.is_playing = False
                    return
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    return
                self.frame_loaded = True
            else:
                # 일시정지 중: 현재 프레임 유지
                if not self.frame_loaded:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if not ret:
                        return
                    self.frame_loaded = True
                else:
                    # 이미 로드된 프레임이 있으면 다시 읽기만 함
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if not ret:
                        return
            
            # 비동기 추론: 최신 프레임을 worker에 전달하고, worker가 반환한
            # 최신 감지 결과 스냅샷을 읽어 화면에 표시합니다.
            try:
                with self._infer_lock:
                    # 항상 최신 프레임만 처리하도록 덮어쓰기
                    self._latest_frame_for_infer = frame.copy()
                    # 스냅샷 복사 (None-safe)
                    bboxes, confidences = self._latest_detections
                    bboxes = list(bboxes)
                    confidences = list(confidences)
            except Exception:
                bboxes, confidences = [], []

            # 차량 방향 감지 (프레임 불필요, BBOX만 필요)
            directions = self.vehicle_detector.detect_vehicle_direction(bboxes)

            # 프레임 차이 방식 활성화 체크 (8초 경과 후 - 초반 주차 차량이 PARKED 도달 대기)
            current_time = self.current_frame / self.fps
            if not self.frame_diff_enabled and current_time >= self.frame_diff_start_time:
                self.frame_diff_enabled = True
                print(f"[INFO] 프레임 차이 방식 활성화 ({current_time:.1f}초 경과)")

            # 각 ROI별 차량 확인 (BBOX 방식 + 프레임 차이 방식 병합)
            vehicles_in_roi_bbox = self._check_vehicles_in_rois(bboxes, directions)

            # 프레임 차이 방식은 8초 후부터만 활성화 (초반 주차 차량이 PARKED 도달 대기)
            if self.frame_diff_enabled:
                vehicles_in_roi_diff = self._check_roi_frame_difference(frame)
                # 두 방식 병합: BBOX 방식 AND 프레임 차이 방식 둘 다 만족해야 차량 있음
                vehicles_in_roi = {}
                # 모든 ROI 인덱스 확인 (BBOX 또는 frame diff에서 검출된 모든 ROI)
                all_roi_indices = set(vehicles_in_roi_bbox.keys()) | set(vehicles_in_roi_diff.keys())
                for roi_idx in all_roi_indices:
                    roi_state = self.parking_tracker.get_roi_state(roi_idx)
                    # 이미 PARKED 또는 DETECTING 상태인 ROI는 프레임 차이 체크 생략 (초반 주차 차량 유지)
                    if roi_state == self.parking_tracker.PARKED or roi_state == self.parking_tracker.DETECTING:
                        vehicles_in_roi[roi_idx] = vehicles_in_roi_bbox.get(roi_idx, False)
                    else:
                        # 새로 감지되는 차량(EMPTY 또는 LEAVING)은 AND 조건 적용
                        vehicles_in_roi[roi_idx] = vehicles_in_roi_bbox.get(roi_idx, False) and vehicles_in_roi_diff.get(roi_idx, False)
            else:
                # 8초 이전에는 BBOX 방식만 사용 (초반 주차 차량 감지)
                vehicles_in_roi = vehicles_in_roi_bbox
            
            # 주차 상태 업데이트
            tracking_info = self.parking_tracker.update(vehicles_in_roi)
            parked_count = tracking_info['parked_count']
            
            # UI 업데이트
            total = len(self.app.rectangles)
            empty = total - parked_count
            self.parked_label.config(text=f"주차됨: {parked_count}")
            self.empty_label.config(text=f"빈칸: {empty}")
            
            # ROI 그리기
            self._draw_rois(frame)

            # 차량 바운딩 박스 및 방향 화살표 그리기
            self._draw_bboxes(frame, bboxes, confidences, directions)
            
            # 화면에 표시
            self._display_frame(frame)
            
            # 다음 프레임 처리 예약
            frame_elapsed = (time.time() - frame_start_time) * 1000  # 밀리초
            self.frame_times.append(frame_elapsed)
            
            # 매 30프레임마다 평균 처리 시간 출력 (디버그용)
            if len(self.frame_times) >= 30:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                log_msg = StatisticsFormatter.format_processing_time(self.fps, avg_time, self.update_interval, self.playback_speed)
                print(log_msg)
                self.frame_times = []
            
            # 배속에 따라 interval 동적 조정
            adjusted_interval = max(1, int(self.update_interval / self.playback_speed))
            self.frame.after(adjusted_interval, self.update_frame)
        
        except Exception as e:
            print(f"프레임 업데이트 오류: {e}")
            self._cleanup()
    
    def _check_roi_frame_difference(self, frame):
        """
        ROI 영역의 프레임 차이를 계산하여 차량 존재 여부 판단

        이전 프레임과 현재 프레임의 ROI 영역을 비교하여
        변화가 있으면 차량이 있는 것으로 판단

        Args:
            frame: 현재 프레임

        Returns:
            dict: {roi_idx: bool} ROI별 차량 존재 여부
        """
        import cv2
        import numpy as np

        vehicles_in_roi = {}

        # 첫 프레임이면 모든 ROI를 False로 초기화하고 현재 프레임 저장
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            for roi_idx in range(len(self.app.rectangles)):
                vehicles_in_roi[roi_idx] = False
            return vehicles_in_roi

        # 그레이스케일 변환
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # 각 ROI별로 프레임 차이 계산
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            # ROI 마스크 생성
            mask = np.zeros(gray_current.shape, dtype=np.uint8)
            pts = np.array(roi_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # ROI 영역만 추출
            roi_current = cv2.bitwise_and(gray_current, gray_current, mask=mask)
            roi_prev = cv2.bitwise_and(gray_prev, gray_prev, mask=mask)

            # 프레임 차이 계산
            diff = cv2.absdiff(roi_current, roi_prev)

            # 임계값 적용 (변화가 큰 픽셀만 추출)
            _, thresh = cv2.threshold(diff, self.roi_frame_diff_threshold, 255, cv2.THRESH_BINARY)

            # 변화된 픽셀 개수 계산
            changed_pixels = cv2.countNonZero(thresh)

            # ROI 면적 계산
            roi_area = cv2.countNonZero(mask)

            # 변화 비율 계산
            change_ratio = changed_pixels / roi_area if roi_area > 0 else 0

            # 변화 비율이 임계값 이상이면 차량 있음으로 판단
            vehicles_in_roi[roi_idx] = change_ratio >= self.roi_change_ratio_threshold

        # 현재 프레임을 이전 프레임으로 저장
        self.prev_frame = frame.copy()

        return vehicles_in_roi

    def _check_vehicles_in_rois(self, bboxes, directions=None):
        """
        각 ROI에 차량이 있는지 확인

        Args:
            bboxes: 바운딩 박스 리스트
            directions: 방향 정보 리스트 (선택사항)

        Returns:
            dict: {roi_idx: bool} ROI별 차량 존재 여부
        """
        vehicles_in_roi = {}

        # 방향 정보가 없으면 None 리스트 생성
        if directions is None:
            directions = [None] * len(bboxes)

        # If we have precomputed integrals, use them for O(1) overlap queries
        if hasattr(self, 'roi_integrals') and self.roi_integrals:
            for roi_idx, roi_info in enumerate(self.roi_integrals):
                vehicles_in_roi[roi_idx] = False
                integral = roi_info['integral']
                roi_area = roi_info['roi_area']
                rx1, ry1, rx2, ry2 = roi_info['bbox']

                for bbox in bboxes:
                    # 원본 BBOX 사용 (대각선 조정 제거)
                    x1, y1, x2, y2 = bbox
                    # 빠른 배제: 바운딩 박스가 서로 겹치지 않으면 건너뜀
                    if x2 <= rx1 or x1 >= rx2 or y2 <= ry1 or y1 >= ry2:
                        continue

                    # 교집합 영역 계산 (클램프)
                    ix1 = max(x1, rx1)
                    iy1 = max(y1, ry1)
                    ix2 = min(x2, rx2)
                    iy2 = min(y2, ry2)

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    # integral은 (h+1, w+1) 형태
                    s = integral[iy2 + 1, ix2 + 1] - integral[iy1, ix2 + 1] - integral[iy2 + 1, ix1] + integral[iy1, ix1]
                    bbox_area = (x2 - x1) * (y2 - y1)

                    # ROI 기준 비율: 교집합이 ROI의 몇 %를 차지하는가
                    roi_ratio = (s / roi_area) if roi_area > 0 else 0

                    if roi_ratio >= self.vehicle_detector.OVERLAP_THRESHOLD:
                        vehicles_in_roi[roi_idx] = True
                        break

            return vehicles_in_roi

        # Fallback: 기존 방식 (polygon check)
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            vehicles_in_roi[roi_idx] = False

            for bbox in bboxes:
                # 원본 BBOX 사용 (대각선 조정 제거)
                if self.vehicle_detector.check_vehicle_in_roi(bbox, roi_polygon):
                    vehicles_in_roi[roi_idx] = True
                    break

        return vehicles_in_roi

    def _prepare_roi_integrals(self):
        """ROI 다각형들에 대해 프레임 크기 기준의 마스크 적분 이미지를 미리 계산합니다.

        이 함수는 play 화면 진입 시 한 번 호출되어, 이후 프레임마다 빠르게 겹침을 계산할 수 있게 합니다.
        """
        try:
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except Exception:
            return

        self.roi_integrals = []
        for poly in self.app.rectangles:
            # 정수 좌표로 변환 및 프레임 내부로 클램프
            pts = np.array(poly, dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            # ROI 마스크 생성 (0/1)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)

            # 적분 이미지 계산 (cv2.integral -> shape (h+1, w+1))
            integral = cv2.integral(mask)
            
            # ROI 면적 계산 (Shoelace formula)
            roi_area = abs(sum(poly[i][0] * poly[(i + 1) % 4][1] - 
                              poly[(i + 1) % 4][0] * poly[i][1] 
                              for i in range(4))) / 2

            # 폴리의 바운딩 박스
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))

            self.roi_integrals.append({'integral': integral, 'bbox': (x_min, y_min, x_max, y_max), 'roi_area': roi_area})
    
    def _draw_rois(self, frame):
        """ROI 그리기 (상태에 따른 색상)"""
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            roi_state = self.parking_tracker.get_roi_state(roi_idx)
            color = ColorUtils.get_state_color(roi_state)
            VideoUtils.draw_roi(frame, roi_polygon, color, thickness=2)
    
    def _draw_bboxes(self, frame, bboxes, confidences, directions=None):
        """
        차량 바운딩 박스 및 방향 화살표 그리기

        Args:
            frame: 대상 이미지
            bboxes: 바운딩 박스 리스트
            confidences: 신뢰도 리스트
            directions: 방향 정보 리스트 (선택사항)
        """
        import cv2
        import math

        if directions is None:
            directions = [None] * len(bboxes)

        for bbox, conf, direction_info in zip(bboxes, confidences, directions):
            x1, y1, x2, y2 = bbox

            # BBOX 그리기 (두께 2, 밝은 파란색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # 신뢰도 표시 (우상단)
            conf_text = f"{conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

            # 방향 화살표 그리기
            if direction_info:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                direction = direction_info.get('direction', 'UNKNOWN')

                # 대각선 이동인 경우 조정된 BBOX도 함께 표시 (초록색 점선)
                diagonal_directions = ['NE', 'SE', 'SW', 'NW']
                if direction in diagonal_directions:
                    adjusted_bbox = self.vehicle_detector.adjust_bbox_for_diagonal_movement(bbox, direction)
                    ax1, ay1, ax2, ay2 = adjusted_bbox
                    # 점선 효과 (여러 개의 짧은 선)
                    dash_length = 5
                    for i in range(ax1, ax2, dash_length * 2):
                        cv2.line(frame, (i, ay1), (min(i + dash_length, ax2), ay1), (0, 255, 0), 1)
                        cv2.line(frame, (i, ay2), (min(i + dash_length, ax2), ay2), (0, 255, 0), 1)
                    for i in range(ay1, ay2, dash_length * 2):
                        cv2.line(frame, (ax1, i), (ax1, min(i + dash_length, ay2)), (0, 255, 0), 1)
                        cv2.line(frame, (ax2, i), (ax2, min(i + dash_length, ay2)), (0, 255, 0), 1)

                # 방향별 처리
                if direction == 'STATIONARY':
                    # 정지 상태: 회색 테두리
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, 'STOP', (center_x - 20, center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                elif direction == 'NEW':
                    # 새로 등장: 노란색 테두리
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, 'NEW', (center_x - 20, center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif direction != 'UNKNOWN':
                    # 이동 중: 화살표 및 방향 표시
                    angle = direction_info.get('angle', 0)
                    angle_rad = math.radians(angle)
                    speed = direction_info.get('speed', 0)

                    # 화살표 길이 (속도에 비례)
                    arrow_length = int(max(30, min(80, speed * 2)))
                    end_x = int(center_x + arrow_length * math.cos(angle_rad))
                    end_y = int(center_y + arrow_length * math.sin(angle_rad))

                    # 화살표 색상 (신뢰도 및 속도에 따라)
                    confidence = direction_info.get('confidence', 0)
                    if confidence > 0.7:
                        color = (0, 255, 0)  # 초록색 (높은 신뢰도)
                    elif confidence > 0.4:
                        color = (0, 255, 255)  # 노랑색 (중간 신뢰도)
                    else:
                        color = (0, 165, 255)  # 주황색 (낮은 신뢰도)

                    # 화살표 그리기 (굵게)
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                                   color, thickness=3, tipLength=0.4)

                    # 방향 텍스트 (화살표 옆)
                    text_x = center_x + 15
                    text_y = center_y - 15
                    cv2.putText(frame, direction, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # 속도 정보 (작게, 하단)
                    speed_text = f"{speed:.1f}px/f"
                    cv2.putText(frame, speed_text, (center_x - 30, y2 + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _display_frame(self, frame):
        """프레임을 화면에 표시"""
        # 리사이징
        frame = GeometryUtils.resize_with_ratio(frame)[0]
        
        # BGR -> RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        
        # 라벨 업데이트
        self.video_panel.config(image=img)
        self.video_panel.image = img
        
        # 슬라이더 및 시간 정보 업데이트 (콜백 없이 순수 값만 업데이트)
        self.slider.set(self.current_frame)
        current_time = self._frame_to_time(self.current_frame)
        total_time = self._frame_to_time(self.total_frames)
        self.time_label.config(text=f"{current_time} / {total_time}")

    def _inference_worker(self):
        """Background inference worker that processes the latest frame.

        - Always consumes the most recent frame (drops older frames)
        - Applies half-resolution detection when playback is fast
        - Stores latest detections in `self._latest_detections` under lock
        """
        infer_times = []
        while not self._infer_stop_event.is_set():
            frame_to_proc = None
            with self._infer_lock:
                if self._latest_frame_for_infer is not None:
                    frame_to_proc = self._latest_frame_for_infer
                    # mark as taken (worker will process this copy)
                    self._latest_frame_for_infer = None

            if frame_to_proc is None:
                # no frame to process right now
                time.sleep(0.01)
                continue

            t0 = time.time()
            try:
                # 속도에 따라 감지 해상도 조정
                if self.playback_speed >= 3.0:
                    h, w = frame_to_proc.shape[:2]
                    small = cv2.resize(frame_to_proc, (max(1, w//2), max(1, h//2)))
                    bboxes, confidences = self.vehicle_detector.detect_vehicles(small)
                    # 좌표 복원
                    bboxes = [[x*2, y*2, x2*2, y2*2] for x, y, x2, y2 in bboxes]
                else:
                    bboxes, confidences = self.vehicle_detector.detect_vehicles(frame_to_proc)
            except Exception as e:
                print(f"[Inference] 오류: {e}")
                bboxes, confidences = [], []

            inf_ms = (time.time() - t0) * 1000.0
            infer_times.append(inf_ms)
            if len(infer_times) >= 30:
                avg = sum(infer_times) / len(infer_times)
                log_msg = StatisticsFormatter.format_inference_time(avg, len(infer_times))
                print(log_msg)
                infer_times = []

            # 저장
            with self._infer_lock:
                self._latest_detections = (bboxes, confidences)

    
    def play(self):
        """재생 시작"""
        self.is_playing = True
    
    def pause(self):
        """일시정지"""
        self.is_playing = False
    
    def set_playback_speed(self, speed):
        """배속 설정"""
        self.playback_speed = speed
        self.speed_label.config(text=f"{speed}x")
    
    def _frame_to_time(self, frame_num):
        """프레임 번호를 HH:MM:SS:MS 형식으로 변환"""
        total_ms = int((frame_num / self.fps) * 1000)
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        seconds = (total_ms % 60000) // 1000
        milliseconds = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    
    def _on_slider_release(self, event):
        """슬라이더에서 마우스를 뗄 때만 콜백 (사용자 수동 조작)"""
        value = self.slider.get()
        self.current_frame = int(value)
        self.playback_time = self.current_frame / self.fps  # 누적 시간 동기화
        self.is_playing = False
        self.frame_loaded = False  # 슬라이더 이동 시 새 프레임 로드 강제
    
    def _cleanup(self):
        """비디오 리소스 정리"""
        # stop inference worker
        try:
            if hasattr(self, '_infer_stop_event') and self._infer_stop_event is not None:
                self._infer_stop_event.set()
            if hasattr(self, '_infer_thread') and self._infer_thread is not None:
                self._infer_thread.join(timeout=1.0)
        except Exception:
            pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _cleanup_and_go_main(self):
        """정리 후 메인 화면으로 이동"""
        self._cleanup()
        self.app.show_main()