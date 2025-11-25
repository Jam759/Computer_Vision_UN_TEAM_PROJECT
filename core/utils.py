"""
공통 유틸리티 함수 모음
"""

import cv2
import numpy as np


class GeometryUtils:
    """기하학 관련 공통 함수"""
    
    @staticmethod
    def validate_polygon(polygon, min_area=100):
        """
        ROI 다각형이 유효한지 검증
        
        Args:
            polygon: [(x1, y1), (x2, y2), ...] 다각형
            min_area: 최소 면적 (기본값: 100)
        
        Returns:
            tuple: (유효 여부, 오류 메시지)
        """
        if len(polygon) != 4:
            return False, f"4개의 점이 필요합니다 (현재: {len(polygon)}개)"
        
        # 좌표 유효성 확인
        for i, point in enumerate(polygon):
            if len(point) != 2 or not all(isinstance(p, (int, float)) for p in point):
                return False, f"점 {i}: 유효하지 않은 좌표"
        
        # 면적 계산 (Shoelace formula)
        area = abs(sum(polygon[i][0] * polygon[(i+1)%4][1] - 
                      polygon[(i+1)%4][0] * polygon[i][1] for i in range(4)) / 2)
        
        if area < min_area:
            return False, f"면적이 너무 작습니다 ({area:.0f} < {min_area})"
        
        return True, ""
    
    @staticmethod
    def resize_with_ratio(frame, max_width=1200, max_height=700):
        """
        종횡비를 유지하며 프레임 리사이징
        
        Args:
            frame: 입력 이미지
            max_width: 최대 너비 (기본값: 1200)
            max_height: 최대 높이 (기본값: 700)
        
        Returns:
            tuple: (리사이징된 프레임, 스케일_x, 스케일_y)
        """
        h, w = frame.shape[:2]
        ratio = min(max_width / w, max_height / h)
        
        if ratio >= 1.0:
            return frame, 1.0, 1.0
        
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(frame, (new_w, new_h))
        
        return resized, w / new_w, h / new_h
    
    @staticmethod
    def point_in_polygon(point, polygon):
        """
        Ray casting 알고리즘으로 점이 다각형 내부인지 판단
        
        Args:
            point: (x, y) 좌표
            polygon: [(x1, y1), (x2, y2), ...] 다각형 꼭짓점
        
        Returns:
            bool: 점이 다각형 내부면 True
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def get_bbox_corners(bbox):
        """
        바운딩 박스 [x1, y1, x2, y2]의 4개 꼭짓점 반환
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
        
        Returns:
            list: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        """
        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    @staticmethod
    def calculate_overlap_ratio(bbox, polygon, grid_samples=20):
        """
        바운딩 박스와 다각형의 겹치는 비율 계산 (그리드 샘플링)
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            polygon: ROI 다각형
            grid_samples: 샘플링 그리드 크기 (기본값: 20)
        
        Returns:
            float: 겹침 비율 (0.0 ~ 1.0)
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area == 0:
            return 0.0
        
        overlap_pixels = 0
        step = max(1, (x2 - x1) // grid_samples)
        
        for x in range(x1, x2, step):
            for y in range(y1, y2, step):
                if GeometryUtils.point_in_polygon((x, y), polygon):
                    overlap_pixels += 1
        
        overlap_ratio = min(overlap_pixels / (grid_samples * grid_samples), 1.0)
        return overlap_ratio


class ColorUtils:
    """색상 관련 공통 함수"""
    
    # BGR 색상 (OpenCV 기준)
    COLOR_GREEN = (0, 255, 0)      # 초록색: 빈 주차칸
    COLOR_ORANGE = (0, 165, 255)   # 주황색: 차량 감지 중
    COLOR_LIGHT_GREEN = (0, 255, 165)  # 밝은 초록: 떠나는 중
    COLOR_RED = (0, 0, 255)        # 빨강: 주차됨
    COLOR_BLUE = (255, 0, 0)       # 파랑: 차량 바운딩 박스
    COLOR_WHITE = (255, 255, 255)  # 흰색
    COLOR_BLACK = (0, 0, 0)        # 검정
    
    @staticmethod
    def get_state_color(state):
        """
        상태에 따른 색상 반환
        
        Args:
            state: 상태 값 (0=EMPTY, 1=DETECTING, 2=PARKED, 3=LEAVING)
        
        Returns:
            tuple: BGR 색상 (R, G, B)
        """
        state_colors = {
            0: ColorUtils.COLOR_GREEN,          # EMPTY
            1: ColorUtils.COLOR_ORANGE,         # DETECTING
            2: ColorUtils.COLOR_RED,            # PARKED
            3: ColorUtils.COLOR_LIGHT_GREEN,    # LEAVING
        }
        return state_colors.get(state, ColorUtils.COLOR_GREEN)


class VideoUtils:
    """영상 처리 관련 공통 함수"""
    
    @staticmethod
    def get_video_fps(cap):
        """
        비디오 캡처 객체에서 FPS 가져오기
        
        Args:
            cap: cv2.VideoCapture 객체
        
        Returns:
            int: FPS (실패 시 기본값 30)
        """
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30
    
    @staticmethod
    def draw_roi(frame, roi_polygon, color, thickness=2):
        """
        ROI 다각형을 프레임에 그리기
        
        Args:
            frame: 대상 이미지
            roi_polygon: [(x1, y1), (x2, y2), ...] 다각형
            color: BGR 색상
            thickness: 선 굵기
        """
        for i in range(4):
            p1 = roi_polygon[i]
            p2 = roi_polygon[(i + 1) % 4]
            cv2.line(frame, p1, p2, color, thickness)
    
    @staticmethod
    def draw_bbox(frame, bbox, conf, color=None, thickness=2):
        """
        바운딩 박스와 신뢰도를 프레임에 그리기
        
        Args:
            frame: 대상 이미지
            bbox: [x1, y1, x2, y2] 바운딩 박스
            conf: 신뢰도 값
            color: BGR 색상 (기본값: 파랑)
            thickness: 선 굵기
        """
        if color is None:
            color = ColorUtils.COLOR_BLUE
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
