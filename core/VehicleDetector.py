"""
YOLOv8 기반 차량 검출 모듈
"""

from ultralytics import YOLO
from core.utils import GeometryUtils


class VehicleDetector:
    """YOLO를 사용한 차량 검출 클래스"""
    
    # COCO 데이터셋 차량 클래스
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    CONFIDENCE_THRESHOLD = 0.5       # 신뢰도 임계값
    OVERLAP_THRESHOLD = 0.2          # 30% 겹침 판정 기준
    
    def __init__(self):
        """YOLOv8 Medium 모델 로드 (GPU 우선, CPU 폴백)"""
        self.model = YOLO('yolov8m.pt')
        
        # GPU 사용 가능 여부 확인 및 설정
        try:
            # GPU 우선 사용
            self.device = 0  # GPU:0 (첫 번째 GPU)
            self.model.to('cuda')
            print("✓ GPU(CUDA) 모드로 실행 중입니다.")
        except Exception as e:
            # GPU 없으면 CPU로 폴백
            self.device = 'cpu'
            self.model.to('cpu')
            print(f"⚠ GPU를 사용할 수 없어 CPU 모드로 전환합니다. ({e})")
    
    def detect_vehicles(self, frame):
        """
        프레임에서 차량 검출
        
        Args:
            frame: 입력 이미지
        
        Returns:
            tuple: (바운딩 박스 리스트, 신뢰도 리스트)
                   [[x1, y1, x2, y2], ...], [conf, ...]
        """
        results = self.model(frame, verbose=False, classes=self.VEHICLE_CLASSES, device=self.device)
        
        bboxes = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    
                    if conf > self.CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        bboxes.append(bbox)
                        confidences.append(conf)
        
        return bboxes, confidences
    
    def check_vehicle_in_roi(self, bbox, roi_polygon, threshold=None):
        """
        바운딩 박스의 일부가 ROI에 겹치는지 확인
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            roi_polygon: ROI 다각형 좌표
            threshold: 겹침 비율 임계값 (기본값: OVERLAP_THRESHOLD)
        
        Returns:
            bool: 겹침 비율이 임계값 이상이면 True
        """
        if threshold is None:
            threshold = self.OVERLAP_THRESHOLD
        
        overlap_ratio = GeometryUtils.calculate_overlap_ratio(bbox, roi_polygon)
        return overlap_ratio >= threshold

