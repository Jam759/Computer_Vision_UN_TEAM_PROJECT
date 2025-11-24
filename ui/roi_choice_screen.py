import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from core.rectangle_selector import RectangleSelector


class ROIScreen:
    def __init__(self, app):
        self.app = app

        frame = tk.Frame(app.container, bg=app.bg_primary)
        frame.pack(fill=tk.BOTH, expand=True, pady=20)

        title = tk.Label(frame, text="주차칸 선택 화면", font=("Arial", 28, "bold"),
                        fg=app.accent_cyan, bg=app.bg_primary)
        title.pack(pady=10)

        roi_button = tk.Button(frame, text="주차칸 설정(OpenCV 창)",
                              bg=app.accent_cyan, fg="#000000",
                              font=("Arial", 11, "bold"), padx=20, pady=10,
                              activebackground="#00a8cc", activeforeground="#000000",
                              relief=tk.FLAT, cursor="hand2",
                              command=self.start_roi)
        roi_button.pack(pady=10)

        self.label = tk.Label(frame, text="ROI: 0개", 
                             fg=app.text_secondary, bg=app.bg_primary, font=("Arial", 11))
        self.label.pack()

        # ROI 미리보기 이미지 (테두리)
        self.preview_frame = tk.Frame(frame, bg=app.bg_secondary, highlightbackground=app.accent_cyan,
                                      highlightthickness=2)
        self.preview_frame.pack(pady=15)
        
        self.preview_label = tk.Label(self.preview_frame, bg=app.bg_secondary)
        self.preview_label.pack()

        button_frame = tk.Frame(frame, bg=app.bg_primary)
        button_frame.pack(pady=20)

        play_button = tk.Button(button_frame, text="영상 재생 화면 →",
                               bg=app.accent_cyan, fg="#000000",
                               font=("Arial", 11, "bold"), padx=15, pady=8,
                               activebackground="#00a8cc", activeforeground="#000000",
                               relief=tk.FLAT, cursor="hand2",
                               command=app.show_play)
        play_button.pack(side=tk.LEFT, padx=5)

        back_button = tk.Button(button_frame, text="← 메인으로",
                               bg=app.accent_red, fg="#000000",
                               font=("Arial", 11, "bold"), padx=15, pady=8,
                               activebackground="#ff5252", activeforeground="#000000",
                               relief=tk.FLAT, cursor="hand2",
                               command=app.show_main)
        back_button.pack(side=tk.LEFT, padx=5)

    # ============================
    # ROI 선택
    # ============================
    def start_roi(self):
        if not self.app.video_path:
            self.label.config(text="영상 먼저 선택하세요")
            return

        cap = cv2.VideoCapture(self.app.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        selector = RectangleSelector(frame)
        selector.run()

        self.app.rectangles = selector.rectangles.copy()
        self.label.config(text=f"ROI: {len(self.app.rectangles)}개")

        self.show_preview(frame)

    # ============================
    # ROI 사각형 그린 이미지 미리보기
    # ============================
    def show_preview(self, frame):
        preview = frame.copy()

        for rect in self.app.rectangles:
            pts = np.array(rect, np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], True, (0, 255, 0), 2)

        preview = self.resize_keep_ratio(preview)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.preview_label.config(image=img)
        self.preview_label.image = img

    def resize_keep_ratio(self, img, max_w=500, max_h=350):
        h, w = img.shape[:2]
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(img, (new_w, new_h))
