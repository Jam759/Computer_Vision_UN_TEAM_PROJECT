import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """YOLO를 사용한 바퀴 검출 클래스"""
    
    def __init__(self):
        # YOLOv8 모델 로드 (바퀴 감지)
        self.model = YOLO('yolov8n.pt')  # nano 모델 (빠른 속도)
        self.wheel_class = 48  # COCO dataset: sports ball (대체용 - 실제로는 wheel 모델 권장)
    
    def detect_wheels(self, frame):
        """
        프레임에서 바퀴를 검출하고 바운딩 박스 반환
        
        Args:
            frame: 입력 영상
        
        Returns:
            list: [x1, y1, x2, y2] 형식의 바운딩 박스 리스트
            list: 각 바운딩 박스의 신뢰도 리스트
        """
        results = self.model(frame, verbose=False, classes=[2, 3, 5, 7])  # 자동차, 오토바이, 버스, 트럭만
        
        bboxes = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        bboxes.append(bbox)
                        confidences.append(conf)
        
        return bboxes, confidences
    
    def get_bbox_corners(self, bbox):
        """
        바운딩 박스의 4개 꼭짓점 반환
        
        Args:
            bbox: [x1, y1, x2, y2] 형식의 바운딩 박스
        
        Returns:
            list: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)] 꼭짓점 리스트
        """
        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def point_in_polygon(self, point, polygon):
        """
        점이 다각형 내부에 있는지 판단 (Ray casting algorithm)
        
        Args:
            point: (x, y) 점
            polygon: [(x1, y1), (x2, y2), ...] 다각형 꼭짓점 리스트
        
        Returns:
            bool: 점이 다각형 내부에 있으면 True
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
    
    def count_corners_in_roi(self, bbox, roi_polygon):
        """
        바퀴 바운딩 박스의 꼭짓점 중 몇 개가 ROI 영역에 들어가 있는지 확인
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            roi_polygon: [(x1, y1), (x2, y2), ...] ROI 다각형
        
        Returns:
            int: ROI에 들어가있는 꼭짓점 개수
        """
        corners = self.get_bbox_corners(bbox)
        count = 0
        
        for corner in corners:
            if self.point_in_polygon(corner, roi_polygon):
                count += 1
        
        return count
    
    def calculate_bbox_overlap_ratio(self, bbox, roi_polygon):
        """
        YOLO 바운딩 박스가 ROI 영역과 겹치는 비율 계산
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            roi_polygon: [(x1, y1), (x2, y2), ...] ROI 다각형
        
        Returns:
            float: 겹치는 영역의 비율 (0.0 ~ 1.0)
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area == 0:
            return 0.0
        
        # 바운딩 박스 내의 픽셀 중 ROI에 들어가있는 픽셀 개수 계산
        overlap_pixels = 0
        
        # 그리드로 샘플링하여 계산 (효율성 향상)
        step = max(1, (x2 - x1) // 20)  # 가로로 최대 20개 샘플
        
        for x in range(x1, x2, step):
            for y in range(y1, y2, step):
                if self.point_in_polygon((x, y), roi_polygon):
                    overlap_pixels += 1
        
        # 바운딩 박스 크기에 따른 정규화
        overlap_ratio = min(overlap_pixels / (20 * 20), 1.0)
        return overlap_ratio
    
    def check_wheel_in_roi_by_percentage(self, bbox, roi_polygon, threshold=0.2):
        """
        YOLO 박스의 30% 이상이 ROI 영역에 들어가있는지 확인
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            roi_polygon: [(x1, y1), (x2, y2), ...] ROI 다각형
            threshold: 필요한 최소 겹침 비율 (기본값: 0.3 = 30%)
        
        Returns:
            bool: 박스의 30% 이상이 ROI에 들어가있으면 True
        """
        overlap_ratio = self.calculate_bbox_overlap_ratio(bbox, roi_polygon)
        return overlap_ratio >= threshold

