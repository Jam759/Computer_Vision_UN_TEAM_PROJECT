"""
메인 화면 - 영상 선택 및 시스템 시작
"""

import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from core.utils import GeometryUtils


class MainScreen:
    """영상 선택 화면"""
    
    # 썸네일 크기
    THUMBNAIL_WIDTH = 450
    THUMBNAIL_HEIGHT = 300
    
    def __init__(self, app):
        """
        Args:
            app: ParkingApp 인스턴스
        """
        self.app = app
        self._build_ui()
    
    def _build_ui(self):
        """UI 구성"""
        frame = tk.Frame(self.app.container, bg=self.app.bg_primary)
        frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 제목
        tk.Label(frame, text="메인 화면", font=("Arial", 28, "bold"),
                fg=self.app.accent_cyan, bg=self.app.bg_primary).pack(pady=10)
        
        # 영상 선택 버튼
        tk.Button(frame, text="영상 선택", command=self.select_file,
                 bg=self.app.accent_cyan, fg="#000000",
                 font=("Arial", 11, "bold"), padx=20, pady=10,
                 activebackground="#00a8cc", relief=tk.FLAT,
                 cursor="hand2").pack(pady=10)
        
        # 파일 경로 표시
        self.label = tk.Label(frame, text="선택된 파일: 없음",
                             fg=self.app.text_secondary, bg=self.app.bg_primary,
                             font=("Arial", 10))
        self.label.pack(pady=5)
        
        # 썸네일 표시 영역
        thumb_frame = tk.Frame(frame, bg=self.app.bg_secondary,
                              highlightbackground=self.app.accent_cyan, highlightthickness=2)
        thumb_frame.pack(pady=15)
        
        self.thumb_label = tk.Label(thumb_frame, bg=self.app.bg_secondary)
        self.thumb_label.pack()
        
        # 다음 단계 버튼
        tk.Button(frame, text="주차칸 선택 화면으로 이동",
                 bg=self.app.accent_cyan, fg="#000000",
                 font=("Arial", 11, "bold"), padx=20, pady=10,
                 activebackground="#00a8cc", relief=tk.FLAT,
                 cursor="hand2", command=self.app.show_roi).pack(pady=30)
    
    def select_file(self):
        """영상 파일 선택 및 썸네일 표시"""
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return
        
        try:
            self.app.video_path = path
            self.label.config(text=f"선택됨: {path.split('/')[-1]}")
            
            # 첫 프레임 썸네일 생성
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise Exception("영상을 열 수 없습니다")
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception("프레임을 읽을 수 없습니다")
            
            # 리사이징 및 표시
            frame = GeometryUtils.resize_with_ratio(
                frame, self.THUMBNAIL_WIDTH, self.THUMBNAIL_HEIGHT)[0]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.thumb_label.config(image=img)
            self.thumb_label.image = img
        
        except Exception as e:
            self.label.config(text=f"❌ 오류: {str(e)}")
            self.app.video_path = None
            print(f"파일 선택 오류: {e}")
