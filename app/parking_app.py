import tkinter as tk
from ui.main_screen import MainScreen
from ui.roi_choice_screen import ROIScreen
from ui.play_screen import PlayScreen

class ParkingApp:
    def __init__(self, root):
        self.root = root
        root.title("Parking Monitor System")
        root.geometry("900x650")
        
        # 밝은 회색 모던 테마 색상
        self.bg_primary = "#f5f5f5"    # 밝은 회색 배경
        self.bg_secondary = "#e8e8e8"  # 보조 배경
        self.accent_cyan = "#0088cc"   # 파란색 악센트
        self.accent_red = "#cc3333"    # 빨강 악센트
        self.text_primary = "#333333"  # 주 텍스트
        self.text_secondary = "#666666" # 보조 텍스트
        
        root.config(bg=self.bg_primary)

        # 공유 데이터
        self.video_path = None
        self.rectangles = []

        # 현재 프레임
        self.container = tk.Frame(root, bg=self.bg_primary)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.show_main()

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    # ------- 화면 전환 함수 -------
    def show_main(self):
        self.clear()
        MainScreen(self)

    def show_roi(self):
        self.clear()
        ROIScreen(self)

    def show_play(self):
        self.clear()
        PlayScreen(self)

if __name__ == "__main__":
    root = tk.Tk()
    ParkingApp(root)
    root.mainloop()
