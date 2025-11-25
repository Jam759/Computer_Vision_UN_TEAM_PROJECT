"""
주차 모니터링 애플리케이션 메인 진입점
"""

import tkinter as tk
from ui.main_screen import MainScreen
from ui.roi_choice_screen import ROIScreen
from ui.play_screen import PlayScreen


class ParkingApp:
    """
    주차장 모니터링 시스템 메인 애플리케이션
    
    화면 구성:
    1. MainScreen: 영상 파일 선택
    2. ROIScreen: 주차칸(ROI) 설정
    3. PlayScreen: 실시간 주차 감지 및 모니터링
    """
    
    # 라이트 그레이 모던 테마 색상
    BG_PRIMARY = "#f5f5f5"      # 주 배경색 (밝은 회색)
    BG_SECONDARY = "#e8e8e8"    # 보조 배경색
    ACCENT_CYAN = "#0088cc"     # 주 강조색 (파랑)
    ACCENT_RED = "#cc3333"      # 보조 강조색 (빨강)
    TEXT_PRIMARY = "#333333"    # 주 텍스트색 (어두운 회색)
    TEXT_SECONDARY = "#666666"  # 보조 텍스트색 (중간 회색)
    
    def __init__(self, root):
        """
        Args:
            root: tkinter.Tk 인스턴스
        """
        self.root = root
        root.title("Parking Monitor System")
        root.geometry("900x650")
        
        # 테마 색상 설정
        self.bg_primary = self.BG_PRIMARY
        self.bg_secondary = self.BG_SECONDARY
        self.accent_cyan = self.ACCENT_CYAN
        self.accent_red = self.ACCENT_RED
        self.text_primary = self.TEXT_PRIMARY
        self.text_secondary = self.TEXT_SECONDARY
        
        root.config(bg=self.bg_primary)
        
        # 공유 데이터
        self.video_path = None      # 선택된 영상 파일 경로
        self.rectangles = []        # ROI 좌표 리스트
        
        # 화면 전환용 컨테이너
        self.container = tk.Frame(root, bg=self.bg_primary)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # 초기 화면 표시
        self.show_main()
    
    def clear(self):
        """현재 표시된 화면 전체 제거"""
        for widget in self.container.winfo_children():
            widget.destroy()
    
    def show_main(self):
        """메인 화면으로 이동"""
        self.clear()
        MainScreen(self)
    
    def show_roi(self):
        """ROI 선택 화면으로 이동"""
        self.clear()
        ROIScreen(self)
    
    def show_play(self):
        """재생 화면으로 이동"""
        self.clear()
        PlayScreen(self)
    
    def cleanup(self):
        """애플리케이션 종료 시 정리"""
        # PlayScreen이 활성화 중이면 리소스 정리
        for widget in self.container.winfo_children():
            if hasattr(widget, 'winfo_class') and 'Frame' in widget.winfo_class():
                # 모든 위젯의 cleanup 메서드 호출 시도
                for child in widget.winfo_children():
                    if hasattr(child, '_cleanup'):
                        try:
                            child._cleanup()
                        except:
                            pass
