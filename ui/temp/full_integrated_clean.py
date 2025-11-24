import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

class FullIntegratedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Monitor System")
        self.root.geometry("1400x900")

        self.selected_video_path = ""
        self.current_points = []
        self.final_rectangles = []
        self.original_frame = None
        self.clean_original_frame = None
        self.image_panel = None

        self.is_playing = False
        self.video_cap = None
        self.delay = 33
        self.after_id = None

        nav = tk.Frame(root)
        nav.pack(fill=tk.X, pady=5)
        ttk.Button(nav, text="영역 지정 / 재생", command=self.show_region_screen).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav, text="실시간 모니터링", command=self.show_live_screen).pack(side=tk.LEFT, padx=5)

        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)

        root.bind('<Key-z>', self.undo_last_point)

        self.show_region_screen()

    def clear_container(self):
        for w in self.container.winfo_children():
            w.destroy()

    def stop_stream(self):
        self.is_playing = False
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None

    def update_image_panel(self, panel, frame, max_width=700):
        if frame is None or panel is None:
            return
        h, w = frame.shape[:2]
        if w > max_width:
            ratio = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * ratio)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel.config(image=img)
        panel.image = img

    def draw_rects(self, frame, rects, color):
        for r in rects:
            pts = np.array(r, np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, color, 2)

    # -------------------- 영역 지정 화면 --------------------
    def show_region_screen(self):
        self.stop_stream()
        self.clear_container()

        frame = tk.Frame(self.container)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="영역 지정", font=("Arial",20)).pack(pady=10)
        ttk.Button(frame, text="영상 선택", command=self.open_video).pack(pady=10)

        self.status_label = tk.Label(frame, text="영상 파일을 선택하세요.", fg="blue")
        self.status_label.pack()

        self.info_label = tk.Label(frame, text="", fg="darkgreen")
        self.info_label.pack()

        btns = tk.Frame(frame)
        btns.pack(pady=10)

        self.play_button = ttk.Button(btns, text="재생", command=self.start_stream, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=10)

        ttk.Button(btns, text="재지정", command=self.reset_selection).pack(side=tk.LEFT, padx=10)

        self.image_panel = tk.Label(frame)
        self.image_panel.pack(pady=10)

        self.region_frame = frame

    def open_video(self):
        self.stop_stream()
        path = filedialog.askopenfilename(
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")]
        )
        if not path:
            return
        self.video_cap = cv2.VideoCapture(path)
        ret, frame = self.video_cap.read()
        if not ret:
            return
        self.clean_original_frame = frame.copy()
        self.original_frame = frame.copy()
        self.current_points = []
        self.final_rectangles = []
        self.status_label.config(text=f"선택됨: {path}", fg="green")
        self.info_label.config(text="이미지 위에 4점을 찍어 사각형 지정 (Z키로 취소)")
        self.update_image_panel(self.image_panel, frame)
        self.image_panel.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.original_frame is None:
            return
        x, y = event.x, event.y
        self.current_points.append((x,y))

        frame = self.clean_original_frame.copy()
        self.draw_rects(frame, self.final_rectangles, (0,0,255))

        for px, py in self.current_points:
            cv2.circle(frame, (px,py), 5, (0,255,0), -1)

        if len(self.current_points) == 4:
            self.final_rectangles.append(self.current_points.copy())
            self.current_points = []
            self.original_frame = self.clean_original_frame.copy()
            self.draw_rects(self.original_frame, self.final_rectangles, (0,0,255))
            self.update_image_panel(self.image_panel, self.original_frame)
            self.play_button.config(state=tk.NORMAL)
            return

        self.update_image_panel(self.image_panel, frame)

    def undo_last_point(self, event=None):
        if not self.current_points:
            return
        self.current_points.pop()
        frame = self.clean_original_frame.copy()
        self.draw_rects(frame, self.final_rectangles, (0,0,255))
        for px, py in self.current_points:
            cv2.circle(frame, (px,py), 5, (0,255,0), -1)
        self.update_image_panel(self.image_panel, frame)

    def reset_selection(self):
        self.stop_stream()
        if self.clean_original_frame is None:
            return
        self.current_points = []
        self.final_rectangles = []
        self.update_image_panel(self.image_panel, self.clean_original_frame)

    def start_stream(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return
        self.is_playing = True
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.run_stream()

    def run_stream(self):
        if not self.is_playing:
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.is_playing = False
            return
        if self.final_rectangles:
            self.draw_rects(frame, self.final_rectangles, (0,255,0))
        self.update_image_panel(self.image_panel, frame)
        self.after_id = self.root.after(self.delay, self.run_stream)

    # -------------------- Live Monitor 화면 --------------------
    def show_live_screen(self):
        self.stop_stream()
        self.clear_container()

        frame = tk.Frame(self.container)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="실시간 모니터링", font=("Arial",20)).pack(pady=10)

        stat = tk.Frame(frame)
        stat.pack()
        self.total_label = tk.Label(stat, text="Total: 0"); self.total_label.pack(side=tk.LEFT, padx=5)
        self.used_label = tk.Label(stat, text="Used: 0"); self.used_label.pack(side=tk.LEFT, padx=5)
        self.empty_label = tk.Label(stat, text="Empty: 0"); self.empty_label.pack(side=tk.LEFT, padx=5)

        self.monitor_panel = tk.Label(frame)
        self.monitor_panel.pack(pady=10)

        log_frame = tk.Frame(frame)
        log_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(log_frame, text="Log").pack()
        self.log_box = tk.Text(log_frame, width=40, height=25)
        self.log_box.pack()

        ctrl = tk.Frame(frame)
        ctrl.pack(pady=10)
        ttk.Button(ctrl, text="Pause/Resume", command=self.toggle_live_pause).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=10)
        tk.Label(ctrl, text="Threshold").pack(side=tk.LEFT)
        self.thr_var = tk.DoubleVar(value=0.5)
        ttk.Scale(ctrl, from_=0, to=1, orient="horizontal", variable=self.thr_var, length=200).pack(side=tk.LEFT, padx=10)

        self.live_paused = False
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.run_live()
        else:
            self.log_box.insert(tk.END, "영역 지정 화면에서 먼저 영상을 선택하세요.\n")

        self.live_frame = frame
        self.last_live_frame = None

    def toggle_live_pause(self):
        self.live_paused = not self.live_paused

    def save_snapshot(self):
        if self.last_live_frame is None:
            return
        filename = f"snapshot_{time.strftime('%H%M%S')}.png"
        cv2.imwrite(filename, self.last_live_frame)
        self.log_box.insert(tk.END, f"Saved: {filename}\n")

    def run_live(self):
        if self.live_frame is None:
            return
        if self.live_paused:
            self.after_id = self.root.after(self.delay, self.run_live)
            return

        ret, frame = self.video_cap.read() if self.video_cap else (False, None)
        if not ret:
            return

        if self.final_rectangles:
            self.draw_rects(frame, self.final_rectangles, (0,255,0))

        self.last_live_frame = frame.copy()
        self.update_image_panel(self.monitor_panel, frame, max_width=900)

        self.after_id = self.root.after(self.delay, self.run_live)


if __name__ == "__main__":
    root = tk.Tk()
    FullIntegratedGUI(root)
    root.mainloop()
