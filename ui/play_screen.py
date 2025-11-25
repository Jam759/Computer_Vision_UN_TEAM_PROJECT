"""
영상 재생 및 실시간 주차 감지 화면
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
from core.VehicleDetector import VehicleDetector
from core.ParkingTracker import ParkingTracker
from core.utils import VideoUtils, ColorUtils, GeometryUtils


class PlayScreen:
    """
    비디오 재생 및 실시간 차량 감지, 주차 상태 시각화
    """
    
    # 재생 설정 (안정성 향상)
    PARKING_SECONDS = 3      # 주차 인정 시간 (초) - 안정성 높음
    UNPARKING_SECONDS = 5    # 비워짐 인정 시간 (초) - 안정성 높음
    UPDATE_INTERVAL = 33     # 프레임 업데이트 주기 (33ms ≈ 30fps)
    
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
            self.is_playing = True
            
            # FPS 정보
            self.fps = VideoUtils.get_video_fps(self.cap)
            
            # 차량 감지기 및 주차 추적기
            self.vehicle_detector = VehicleDetector()
            self.parking_tracker = ParkingTracker(
                fps=self.fps,
                parking_seconds=self.PARKING_SECONDS,
                unparking_seconds=self.UNPARKING_SECONDS,
                detection_tolerance_seconds=1  # YOLO 감지 실패 1초 허용
            )
            
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
        header_frame = tk.Frame(self.frame, bg=self.app.bg_primary)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 네비게이션 버튼
        nav_frame = tk.Frame(header_frame, bg=self.app.bg_primary)
        nav_frame.pack(side=tk.LEFT)
        
        tk.Button(nav_frame, text="← 주차칸 다시 설정",
                 bg=self.app.accent_cyan, fg="#000000",
                 font=("Arial", 10, "bold"), padx=6, pady=6,
                 activebackground="#0088cc",
                 relief=tk.FLAT, cursor="hand2",
                 command=self.app.show_roi).pack(side=tk.LEFT, padx=3)
        
        tk.Button(nav_frame, text="← 메인",
                 bg=self.app.accent_red, fg="#000000",
                 font=("Arial", 10, "bold"), padx=6, pady=6,
                 activebackground="#cc0000",
                 relief=tk.FLAT, cursor="hand2",
                 command=self.app.show_main).pack(side=tk.LEFT, padx=3)
        
        # 제목
        tk.Label(header_frame, text="영상 재생 화면",
                font=("Arial", 28, "bold"),
                fg=self.app.accent_cyan, bg=self.app.bg_primary).pack(side=tk.LEFT, expand=True)
    
    def _build_info_panel(self):
        """정보 패널 (전체/주차됨/빈칸) 구성"""
        info = tk.Frame(self.frame, bg=self.app.bg_secondary,
                       highlightbackground=self.app.accent_cyan, highlightthickness=1)
        info.pack(pady=5)
        
        total = len(self.app.rectangles)
        
        self.total_label = tk.Label(info, text=f"전체: {total}",
                                   fg=self.app.text_primary, bg=self.app.bg_secondary,
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.total_label.pack(side=tk.LEFT, padx=5)
        
        self.parked_label = tk.Label(info, text="주차됨: 0",
                                    fg=self.app.accent_red, bg=self.app.bg_secondary,
                                    font=("Arial", 11, "bold"), padx=15, pady=8)
        self.parked_label.pack(side=tk.LEFT, padx=5)
        
        self.empty_label = tk.Label(info, text=f"빈칸: {total}",
                                   fg=self.app.accent_cyan, bg=self.app.bg_secondary,
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.empty_label.pack(side=tk.LEFT, padx=5)
    
    def _build_video_panel(self):
        """영상 표시 영역 구성"""
        video_frame = tk.Frame(self.frame, bg=self.app.bg_secondary,
                              highlightbackground=self.app.accent_cyan, highlightthickness=2)
        video_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.video_panel = tk.Label(video_frame, bg=self.app.bg_secondary)
        self.video_panel.pack(fill=tk.BOTH, expand=True)
    
    def _build_error_ui(self, error_message):
        """에러 UI 구성"""
        self.frame = tk.Frame(self.app.container, bg=self.app.bg_primary)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 에러 메시지
        tk.Label(self.frame, text="❌ 오류 발생", 
                font=("Arial", 24, "bold"),
                fg=self.app.accent_red, bg=self.app.bg_primary).pack(pady=20)
        
        tk.Label(self.frame, text=error_message,
                font=("Arial", 12), fg=self.app.text_secondary,
                bg=self.app.bg_primary, wraplength=400).pack(pady=10)
        
        # 돌아가기 버튼
        tk.Button(self.frame, text="← 메인으로 돌아가기",
                 bg=self.app.accent_red, fg="#000000",
                 font=("Arial", 11, "bold"), padx=20, pady=10,
                 activebackground="#cc0000", relief=tk.FLAT,
                 cursor="hand2", command=self._cleanup_and_go_main).pack(pady=20)
    
    def _build_control_bar(self):
        """재생 제어 바 구성"""
        control_frame = tk.Frame(self.frame, bg=self.app.bg_primary)
        control_frame.pack(pady=5, fill=tk.X, padx=10)
        
        # 재생/일시정지 버튼
        tk.Button(control_frame, text="▶ 재생/일시정지",
                 bg=self.app.accent_cyan, fg="#000000",
                 font=("Arial", 10, "bold"), padx=10, pady=5,
                 activebackground="#0088cc",
                 relief=tk.FLAT, cursor="hand2",
                 command=self.toggle_play).pack(side=tk.LEFT, padx=5)
        
        # 슬라이더
        slider_frame = tk.Frame(control_frame, bg=self.app.bg_primary)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(slider_frame, text="위치:", fg=self.app.text_primary,
                bg=self.app.bg_primary, font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.slider = tk.Scale(slider_frame, from_=0, to=self.total_frames-1,
                              orient=tk.HORIZONTAL,
                              bg=self.app.bg_secondary, fg=self.app.text_primary,
                              highlightbackground=self.app.accent_cyan,
                              command=self.seek_frame, length=200)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.frame_label = tk.Label(slider_frame, text=f"0/{self.total_frames}",
                                   fg=self.app.text_secondary, bg=self.app.bg_primary,
                                   font=("Arial", 9))
        self.frame_label.pack(side=tk.LEFT, padx=5)
    
    def update_frame(self):
        """각 프레임 처리 및 업데이트"""
        try:
            # 프레임 읽기
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    return
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    return
            
            # 차량 감지
            bboxes, confidences = self.vehicle_detector.detect_vehicles(frame)
            
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
            self.frame.after(self.UPDATE_INTERVAL, self.update_frame)
        
        except Exception as e:
            print(f"프레임 업데이트 오류: {e}")
            self._cleanup()
    
    def _check_vehicles_in_rois(self, bboxes):
        """각 ROI에 차량이 있는지 확인"""
        vehicles_in_roi = {}
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            vehicles_in_roi[roi_idx] = False
            
            for bbox in bboxes:
                if self.vehicle_detector.check_vehicle_in_roi(bbox, roi_polygon):
                    vehicles_in_roi[roi_idx] = True
                    break
        
        return vehicles_in_roi
    
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
        
        # 슬라이더 및 프레임 정보 업데이트
        self.slider.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")
    
    def toggle_play(self):
        """재생/일시정지 토글"""
        self.is_playing = not self.is_playing
    
    def seek_frame(self, value):
        """프레임 위치 변경"""
        self.current_frame = int(value)
        self.is_playing = False
    
    def _cleanup(self):
        """비디오 리소스 정리"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _cleanup_and_go_main(self):
        """정리 후 메인 화면으로 이동"""
        self._cleanup()
        self.app.show_main()