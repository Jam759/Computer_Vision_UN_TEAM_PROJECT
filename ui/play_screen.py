import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from core.VehicleDetector import VehicleDetector
from core.ParkingTracker import ParkingTracker


class PlayScreen:
    def __init__(self, app):
        self.app = app
        self.cap = cv2.VideoCapture(app.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.is_playing = True

        # 영상의 FPS 정보 가져오기
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30  # 기본값: 30fps

        # YOLO 차량 검출기 및 주차 추적기 초기화 (FPS 기반)
        self.vehicle_detector = VehicleDetector()
        self.parking_tracker = ParkingTracker(fps=self.fps, parking_seconds=30, unparking_seconds=10)

        self.frame = tk.Frame(app.container, bg=app.bg_primary)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 상단 헤더 프레임 (네비게이션 + 타이틀)
        header_frame = tk.Frame(self.frame, bg=app.bg_primary)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # 왼쪽: 네비게이션 버튼
        nav_frame = tk.Frame(header_frame, bg=app.bg_primary)
        nav_frame.pack(side=tk.LEFT)

        roi_button = tk.Button(nav_frame, text="← 주차칸 다시 설정",
                              bg=app.accent_cyan, fg="#000000",
                              font=("Arial", 10, "bold"), padx=6, pady=6,
                              activebackground="#0088cc",
                              relief=tk.FLAT, cursor="hand2",
                              command=app.show_roi)
        roi_button.pack(side=tk.LEFT, padx=3)

        main_button = tk.Button(nav_frame, text="← 메인",
                               bg=app.accent_red, fg="#000000",
                               font=("Arial", 10, "bold"), padx=6, pady=6,
                               activebackground="#cc0000",
                               relief=tk.FLAT, cursor="hand2",
                               command=app.show_main)
        main_button.pack(side=tk.LEFT, padx=3)

        # 중앙: 타이틀
        title = tk.Label(header_frame, text="영상 재생 화면", font=("Arial", 28, "bold"),
                        fg=app.accent_cyan, bg=app.bg_primary)
        title.pack(side=tk.LEFT, expand=True)

        # 정보 라벨 (테두리)
        info = tk.Frame(self.frame, bg=app.bg_secondary, highlightbackground=app.accent_cyan,
                       highlightthickness=1)
        info.pack(pady=5)

        total = len(app.rectangles)
        self.parked_count = 0
        empty = total - self.parked_count

        self.total_label = tk.Label(info, text=f"전체: {total}", 
                                   fg=app.text_primary, bg=app.bg_secondary, 
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.total_label.pack(side=tk.LEFT, padx=5)

        self.parked_label = tk.Label(info, text=f"주차됨: {self.parked_count}", 
                                    fg=app.accent_red, bg=app.bg_secondary,
                                    font=("Arial", 11, "bold"), padx=15, pady=8)
        self.parked_label.pack(side=tk.LEFT, padx=5)

        self.empty_label = tk.Label(info, text=f"빈칸: {empty}", 
                                   fg=app.accent_cyan, bg=app.bg_secondary,
                                   font=("Arial", 11, "bold"), padx=15, pady=8)
        self.empty_label.pack(side=tk.LEFT, padx=5)

        # 영상 표시 (테두리)
        video_frame = tk.Frame(self.frame, bg=app.bg_secondary, highlightbackground=app.accent_cyan,
                              highlightthickness=2)
        video_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.video_panel = tk.Label(video_frame, bg=app.bg_secondary)
        self.video_panel.pack(fill=tk.BOTH, expand=True)

        # 재생 제어 바
        control_frame = tk.Frame(self.frame, bg=app.bg_primary)
        control_frame.pack(pady=5, fill=tk.X, padx=10)

        play_button = tk.Button(control_frame, text="▶ 재생/일시정지",
                               bg=app.accent_cyan, fg="#000000",
                               font=("Arial", 10, "bold"), padx=10, pady=5,
                               activebackground="#0088cc",
                               relief=tk.FLAT, cursor="hand2",
                               command=self.toggle_play)
        play_button.pack(side=tk.LEFT, padx=5)

        # 슬라이더
        slider_frame = tk.Frame(control_frame, bg=app.bg_primary)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(slider_frame, text="위치:", fg=app.text_primary, bg=app.bg_primary,
                font=("Arial", 10)).pack(side=tk.LEFT)

        self.slider = tk.Scale(slider_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL,
                              bg=app.bg_secondary, fg=app.text_primary,
                              highlightbackground=app.accent_cyan,
                              command=self.seek_frame, length=200)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.frame_label = tk.Label(slider_frame, text=f"0/{self.total_frames}",
                                   fg=app.text_secondary, bg=app.bg_primary,
                                   font=("Arial", 9))
        self.frame_label.pack(side=tk.LEFT, padx=5)

        self.update_frame()

    # ============================
    # 비율 유지 resize
    # ============================
    def resize_keep_ratio(self, frame):
        panel_w = 1200
        panel_h = 700

        h, w = frame.shape[:2]
        ratio = min(panel_w / w, panel_h / h)

        new_w = int(w * ratio)
        new_h = int(h * ratio)

        return cv2.resize(frame, (new_w, new_h))

    # ============================
    # 영상 업데이트 (루프)
    # ============================
    def update_frame(self):
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

        # 바퀴 검출
        bboxes, confidences = self.vehicle_detector.detect_wheels(frame)
        
        # 각 ROI별 바퀴 확인 (박스의 30% 이상이 ROI에 들어와야 함)
        wheels_in_roi = {}
        for roi_idx, roi_polygon in enumerate(self.app.rectangles):
            wheels_in_roi[roi_idx] = False
            
            # 이 ROI에 박스의 30% 이상이 들어가는 바퀴가 있는지 확인
            for bbox in bboxes:
                if self.vehicle_detector.check_wheel_in_roi_by_percentage(bbox, roi_polygon, threshold=0.3):
                    wheels_in_roi[roi_idx] = True
                    break
        
        # 주차 추적 업데이트
        tracking_info = self.parking_tracker.update(wheels_in_roi)
        self.parked_count = tracking_info['parked_count']
        
        # UI 라벨 업데이트
        total = len(self.app.rectangles)
        empty = total - self.parked_count
        self.parked_label.config(text=f"주차됨: {self.parked_count}")
        self.empty_label.config(text=f"빈칸: {empty}")

        # ROI 그리기
        for roi_idx, rect in enumerate(self.app.rectangles):
            for i in range(4):
                p1 = rect[i]
                p2 = rect[(i + 1) % 4]
                
                # 주차된 ROI는 빨강, 아니면 초록색
                if roi_idx in self.parking_tracker.parked_vehicles:
                    color = (0, 0, 255)  # 빨강
                else:
                    color = (0, 255, 0)  # 초록색
                
                cv2.line(frame, p1, p2, color, 2)
        
        # 차량 바운딩 박스 그리기
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # 신뢰도 표시
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame = self.resize_keep_ratio(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self.video_panel.config(image=img)
        self.video_panel.image = img

        self.slider.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")

        self.frame.after(33, self.update_frame)

    # ============================
    # 재생/일시정지 토글
    # ============================
    def toggle_play(self):
        self.is_playing = not self.is_playing

    # ============================
    # 프레임 이동
    # ============================
    def seek_frame(self, value):
        self.current_frame = int(value)
        self.is_playing = False
