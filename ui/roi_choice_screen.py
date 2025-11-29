"""
주차칸 선택 화면 - ROI 설정 및 미리보기
"""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from core.rectangle_selector import RectangleSelector
from core.utils import GeometryUtils
from core.ui_components import UIStyle, UIButton, UILabel, UIFrame


class ROIScreen:
    """ROI(주차칸) 선택 및 관리 화면"""
    
    # 미리보기 크기
    PREVIEW_WIDTH = 500
    PREVIEW_HEIGHT = 350
    
    def __init__(self, app):
        """
        Args:
            app: ParkingApp 인스턴스
        """
        self.app = app
        self._build_ui()
    
    def _build_ui(self):
        """UI 구성"""
        frame = tk.Frame(self.app.container, bg=UIStyle.BG_PRIMARY)
        frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 제목
        UILabel.create_title(frame, "주차칸 선택 화면", UIStyle.BG_PRIMARY).pack(pady=10)
        
        # ROI 선택 버튼
        UIButton.create_primary(
            frame,
            "주차칸 설정(OpenCV 창)",
            self.start_roi,
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # ROI 개수 표시
        self.label = UILabel.create_secondary(frame, "ROI: 0개", UIStyle.BG_PRIMARY)
        self.label.pack()
        
        # 미리보기 영역
        preview_frame = UIFrame.create_video_panel(frame)
        preview_frame.pack(pady=15)
        
        self.preview_label = tk.Label(preview_frame, bg=UIStyle.BG_SECONDARY)
        self.preview_label.pack()
        
        # 네비게이션 버튼
        button_frame = tk.Frame(frame, bg=UIStyle.BG_PRIMARY)
        button_frame.pack(pady=20)
        
        UIButton.create_primary(
            button_frame,
            "영상 재생 화면 →",
            self.app.show_play,
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        UIButton.create_danger(
            button_frame,
            "← 메인으로",
            self.app.show_main,
            padx=15,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
    
    def start_roi(self):
        """ROI 선택 시작"""
        if not self.app.video_path:
            self.label.config(text="❌ 영상 먼저 선택하세요")
            return
        
        try:
            # 영상의 첫 프레임 로드
            cap = cv2.VideoCapture(self.app.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.label.config(text="❌ 영상 로드 실패")
                return
            
            # ROI 선택 창 실행
            selector = RectangleSelector(frame)
            selector.run()
            
            # ROI 유효성 검증
            if len(selector.rectangles) == 0:
                self.label.config(text="✓ ROI: 0개 (선택되지 않음)")
                return
            
            invalid_rois = []
            for idx, roi in enumerate(selector.rectangles):
                valid, error_msg = GeometryUtils.validate_polygon(roi)
                if not valid:
                    invalid_rois.append((idx, error_msg))
            
            if invalid_rois:
                error_text = "❌ 유효하지 않은 ROI:\n"
                for idx, msg in invalid_rois:
                    error_text += f"  ROI {idx}: {msg}\n"
                self.label.config(text=error_text)
                return
            
            # ROI 저장 및 UI 업데이트
            self.app.rectangles = selector.rectangles.copy()
            self.label.config(text=f"✓ ROI: {len(self.app.rectangles)}개 설정됨")
            
            # 미리보기 표시
            self._show_preview(frame)
        
        except Exception as e:
            self.label.config(text=f"❌ 오류: {str(e)}")
            print(f"ROI 선택 오류: {e}")
    
    def _show_preview(self, frame):
        """ROI가 표시된 프레임 미리보기"""
        preview = frame.copy()
        
        # 각 ROI 다각형 그리기
        for rect in self.app.rectangles:
            pts = np.array(rect, np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        
        # 리사이징
        preview = GeometryUtils.resize_with_ratio(
            preview, self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)[0]
        
        # 표시
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.config(image=img)
        self.preview_label.image = img
