"""
공통 UI 컴포넌트 및 스타일 모음
"""

import tkinter as tk


class UIStyle:
    """UI 스타일 및 색상 상수"""
    
    # 라이트 그레이 모던 테마 색상
    BG_PRIMARY = "#f5f5f5"      # 주 배경색 (밝은 회색)
    BG_SECONDARY = "#e8e8e8"    # 보조 배경색
    ACCENT_CYAN = "#0088cc"     # 주 강조색 (파랑)
    ACCENT_RED = "#cc3333"      # 보조 강조색 (빨강)
    TEXT_PRIMARY = "#333333"    # 주 텍스트색 (어두운 회색)
    TEXT_SECONDARY = "#666666"  # 보조 텍스트색 (중간 회색)
    
    # 폰트
    FONT_TITLE = ("Arial", 28, "bold")
    FONT_BUTTON = ("Arial", 11, "bold")
    FONT_LABEL = ("Arial", 11)
    FONT_SMALL = ("Arial", 10)
    FONT_TINY = ("Arial", 9)


class UIButton:
    """공통 버튼 생성 헬퍼"""
    
    @staticmethod
    def create_primary(parent, text, command=None, **kwargs):
        """주 강조색 버튼 (파란색)"""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=UIStyle.ACCENT_CYAN,
            fg="#000000",
            font=UIStyle.FONT_BUTTON,
            activebackground="#0088cc",
            relief=tk.FLAT,
            cursor="hand2",
            **kwargs
        )
    
    @staticmethod
    def create_secondary(parent, text, command=None, **kwargs):
        """보조 버튼 (회색)"""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=UIStyle.BG_SECONDARY,
            fg=UIStyle.TEXT_PRIMARY,
            font=UIStyle.FONT_SMALL,
            relief=tk.FLAT,
            cursor="hand2",
            **kwargs
        )
    
    @staticmethod
    def create_danger(parent, text, command=None, **kwargs):
        """위험 버튼 (빨간색)"""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=UIStyle.ACCENT_RED,
            fg="#000000",
            font=UIStyle.FONT_BUTTON,
            activebackground="#cc0000",
            relief=tk.FLAT,
            cursor="hand2",
            **kwargs
        )
    
    @staticmethod
    def create_speed_button(parent, text, command=None, is_active=False, **kwargs):
        """배속 조절 버튼"""
        bg = UIStyle.ACCENT_CYAN if is_active else UIStyle.BG_SECONDARY
        fg = "#000000" if is_active else UIStyle.TEXT_PRIMARY
        weight = "bold" if is_active else "normal"
        
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            font=("Arial", 9, weight),
            padx=5,
            pady=3,
            relief=tk.FLAT,
            cursor="hand2",
            **kwargs
        )


class UILabel:
    """공통 레이블 생성 헬퍼"""
    
    @staticmethod
    def create_title(parent, text, bg=None):
        """제목 레이블"""
        if bg is None:
            bg = UIStyle.BG_PRIMARY
        return tk.Label(
            parent,
            text=text,
            font=UIStyle.FONT_TITLE,
            fg=UIStyle.ACCENT_CYAN,
            bg=bg
        )
    
    @staticmethod
    def create_default(parent, text, bg=None):
        """기본 레이블"""
        if bg is None:
            bg = UIStyle.BG_PRIMARY
        return tk.Label(
            parent,
            text=text,
            fg=UIStyle.TEXT_PRIMARY,
            bg=bg,
            font=UIStyle.FONT_LABEL
        )
    
    @staticmethod
    def create_secondary(parent, text, bg=None):
        """보조 레이블"""
        if bg is None:
            bg = UIStyle.BG_PRIMARY
        return tk.Label(
            parent,
            text=text,
            fg=UIStyle.TEXT_SECONDARY,
            bg=bg,
            font=UIStyle.FONT_SMALL
        )


class UIFrame:
    """공통 프레임 생성 헬퍼"""
    
    @staticmethod
    def create_panel(parent, with_border=False):
        """패널 프레임"""
        frame = tk.Frame(parent, bg=UIStyle.BG_SECONDARY)
        if with_border:
            frame.config(
                highlightbackground=UIStyle.ACCENT_CYAN,
                highlightthickness=1
            )
        return frame
    
    @staticmethod
    def create_video_panel(parent):
        """영상 표시 패널"""
        return tk.Frame(
            parent,
            bg=UIStyle.BG_SECONDARY,
            highlightbackground=UIStyle.ACCENT_CYAN,
            highlightthickness=2
        )


class StatisticsFormatter:
    """통계 정보 포맷팅"""
    
    @staticmethod
    def format_parking_stats(total, parked, empty):
        """주차 통계 포맷"""
        return {
            'total': f"전체: {total}",
            'parked': f"주차됨: {parked}",
            'empty': f"빈칸: {empty}"
        }
    
    @staticmethod
    def format_processing_time(fps, avg_time, interval, speed):
        """프레임 처리 시간 포맷"""
        return f"[FPS {fps}] 평균 프레임 처리 시간: {avg_time:.2f}ms (설정 interval: {interval}ms) | 배속: {speed}x"
    
    @staticmethod
    def format_inference_time(avg_time, runs):
        """추론 시간 포맷"""
        return f"[Inference] 평균 처리 시간: {avg_time:.2f}ms over {runs} runs"
