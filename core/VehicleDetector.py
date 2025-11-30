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
    OVERLAP_THRESHOLD = 0.10         # ROI 기준 겹침 비율 (BBOX가 ROI의 10% 이상 차지)
    BBOX_SHRINK_RATIO = 0.80         # BBOX 축소 비율 (80% 유지, 20% 제거)
                                      # 대각선 차량의 불필요한 모서리 제거
    
    def __init__(self):
        """YOLOv8 Large 모델 로드 (GPU 우선, CPU 폴백)"""
        self.model = YOLO('yolov8l.pt')
        
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
                        
                        # BBOX 중심 영역만 추출 (대각선 차량의 모서리 제거)
                        bbox = self._shrink_bbox(bbox, self.BBOX_SHRINK_RATIO)
                        
                        bboxes.append(bbox)
                        confidences.append(conf)
        
        return bboxes, confidences
    
    @staticmethod
    def _shrink_bbox(bbox, shrink_ratio=0.80):
        """
        BBOX를 중심 기준으로 축소
        
        대각선으로 보는 차량의 경우 BBOX가 커서 불필요한 모서리가 포함됨
        중심 영역(shrink_ratio)만 유지하여 더 정확한 차량 중심부만 검사
        
        Args:
            bbox: [x1, y1, x2, y2]
            shrink_ratio: 유지할 비율 (0.8 = 80% 유지, 20% 제거)
        
        Returns:
            list: [x1_new, y1_new, x2_new, y2_new]
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 중심에서의 축소 거리
        dx = width * (1 - shrink_ratio) / 2
        dy = height * (1 - shrink_ratio) / 2
        
        x1_new = int(x1 + dx)
        y1_new = int(y1 + dy)
        x2_new = int(x2 - dx)
        y2_new = int(y2 - dy)
        
        return [x1_new, y1_new, x2_new, y2_new]
    
    def check_vehicle_in_roi(self, bbox, roi_polygon, threshold=None):
        """
        BBOX가 ROI의 일정 비율(%)을 차지하는지 확인
        
        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            roi_polygon: ROI 다각형 좌표
            threshold: 임계값 (BBOX가 ROI의 몇 %를 차지해야 하는가) (기본값: OVERLAP_THRESHOLD)
        
        Returns:
            bool: BBOX가 ROI를 임계값 이상 차지하면 True
        """
        if threshold is None:
            threshold = self.OVERLAP_THRESHOLD
        
        # ROI 기준 비율: BBOX가 ROI의 몇 %를 차지하는가
        roi_ratio = GeometryUtils.calculate_bbox_in_roi_ratio(bbox, roi_polygon)
        return roi_ratio >= threshold

