import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk


class MainScreen:
    def __init__(self, app):
        self.app = app

        frame = tk.Frame(app.container, bg=app.bg_primary)
        frame.pack(fill=tk.BOTH, expand=True, pady=20)

        title = tk.Label(frame, text="메인 화면", font=("Arial", 28, "bold"), 
                         fg=app.accent_cyan, bg=app.bg_primary)
        title.pack(pady=10)

        self.select_button = tk.Button(frame, text="영상 선택", command=self.select_file,
                                       bg=app.accent_cyan, fg="#000000", 
                                       font=("Arial", 11, "bold"), padx=20, pady=10,
                                       activebackground="#00a8cc", activeforeground="#000000",
                                       relief=tk.FLAT, cursor="hand2")
        self.select_button.pack(pady=10)

        self.label = tk.Label(frame, text="선택된 파일: 없음", 
                             fg=app.text_secondary, bg=app.bg_primary, font=("Arial", 10))
        self.label.pack(pady=5)

        # 썸네일 표시 영역 (테두리)
        self.thumb_frame = tk.Frame(frame, bg=app.bg_secondary, highlightbackground=app.accent_cyan,
                                    highlightthickness=2)
        self.thumb_frame.pack(pady=15)
        
        self.thumb_label = tk.Label(self.thumb_frame, bg=app.bg_secondary)
        self.thumb_label.pack()

        next_button = tk.Button(frame, text="주차칸 선택 화면으로 이동",
                               bg=app.accent_cyan, fg="#000000",
                               font=("Arial", 11, "bold"), padx=20, pady=10,
                               activebackground="#00a8cc", activeforeground="#000000",
                               relief=tk.FLAT, cursor="hand2",
                               command=app.show_roi)
        next_button.pack(pady=30)

    # ============================
    # 썸네일 비율 유지용 resize 함수
    # ============================
    def resize_keep_ratio(self, img, max_w=450, max_h=300):
        h, w = img.shape[:2]
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(img, (new_w, new_h))

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        self.app.video_path = path
        self.label.config(text=f"선택됨: {path}")

        # 썸네일 생성
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = self.resize_keep_ratio(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.thumb_label.config(image=img)
            self.thumb_label.image = img
