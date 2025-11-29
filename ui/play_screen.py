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
        speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
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
        
        self.frame_label = tk.Label(slider_frame, text=f"0/{self.total_frames}",
                                   fg=UIStyle.TEXT_SECONDARY, bg=UIStyle.BG_PRIMARY,
                                   font=UIStyle.FONT_TINY)
        self.frame_label.pack(side=tk.LEFT, padx=5)
    
    def update_frame(self):
        """각 프레임 처리 및 업데이트"""
        frame_start_time = time.time()
        
        try:
            # 프레임 읽기
            if self.is_playing:
                # 재생 중: 다음 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    # 영상 끝에 도달
                    self.is_playing = False
                    return
                # 현재 프레임 인덱스 수동 증가 (배속에 따라 조정)
                self.current_frame += int(self.playback_speed)
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

            # 각 ROI별 차량 확인
            vehicles_in_roi = self._check_vehicles_in_rois(bboxes)
            
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
            
            # 차량 바운딩 박스 그리기
            self._draw_bboxes(frame, bboxes, confidences)
            
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
    
    def _check_vehicles_in_rois(self, bboxes):
        """각 ROI에 차량이 있는지 확인"""
        vehicles_in_roi = {}

        # If we have precomputed integrals, use them for O(1) overlap queries
        if hasattr(self, 'roi_integrals') and self.roi_integrals:
            for roi_idx, roi_info in enumerate(self.roi_integrals):
                vehicles_in_roi[roi_idx] = False
                integral = roi_info['integral']
                rx1, ry1, rx2, ry2 = roi_info['bbox']

                for bbox in bboxes:
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
                    overlap_ratio = (s / bbox_area) if bbox_area > 0 else 0

                    if overlap_ratio >= self.vehicle_detector.OVERLAP_THRESHOLD:
                        vehicles_in_roi[roi_idx] = True
                        break

            return vehicles_in_roi

        # Fallback: 기존 방식 (polygon check)
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            vehicles_in_roi[roi_idx] = False

            for bbox in bboxes:
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

            # 폴리의 바운딩 박스
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))

            self.roi_integrals.append({'integral': integral, 'bbox': (x_min, y_min, x_max, y_max)})
    
    def _draw_rois(self, frame):
        """ROI 그리기 (상태에 따른 색상)"""
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            roi_state = self.parking_tracker.get_roi_state(roi_idx)
            color = ColorUtils.get_state_color(roi_state)
            VideoUtils.draw_roi(frame, roi_polygon, color, thickness=2)
    
    def _draw_bboxes(self, frame, bboxes, confidences):
        """차량 바운딩 박스 그리기"""
        for bbox, conf in zip(bboxes, confidences):
            VideoUtils.draw_bbox(frame, bbox, conf)
    
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
        
        # 슬라이더 및 프레임 정보 업데이트 (콜백 없이 순수 값만 업데이트)
        self.slider.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")

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
    
    def _on_slider_release(self, event):
        """슬라이더에서 마우스를 뗄 때만 콜백 (사용자 수동 조작)"""
        value = self.slider.get()
        self.current_frame = int(value)
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