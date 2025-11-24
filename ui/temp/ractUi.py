import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
from videoProcess.RectangleManager import RectangleSelector


class RectangleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectangle Manager UI")

        self.video_path = None
        self.frame = None
        self.selector = None

        # UI
        self.load_btn = tk.Button(root, text="Load Video", command=self.load_video, width=20)
        self.load_btn.pack(pady=5)

        self.mode_var = tk.StringVar(value="add")
        tk.Radiobutton(root, text="Add Mode", variable=self.mode_var, value="add", command=self.set_mode).pack()
        tk.Radiobutton(root, text="Delete Mode", variable=self.mode_var, value="delete", command=self.set_mode).pack()

        self.run_btn = tk.Button(root, text="Start Selector", command=self.start_selector, width=20)
        self.run_btn.pack(pady=10)

        self.preview_label = tk.Label(root)
        self.preview_label.pack()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not path:
            return

        self.video_path = path
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to load video frame!")
            return

        self.frame = frame
        self.show_preview(frame)

    def show_preview(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

    def set_mode(self):
        if self.selector:
            self.selector.set_mode(self.mode_var.get())

    def start_selector(self):
        if self.frame is None:
            print("No video loaded!")
            return

        def run_selector():
            self.selector = RectangleSelector(self.frame, mode=self.mode_var.get())
            self.selector.run()

        threading.Thread(target=run_selector, daemon=True).start()
