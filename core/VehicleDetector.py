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

        # 이전 프레임의 차량 위치 저장 (방향 추적용)
        self.prev_bboxes = []
    
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
    
    def detect_vehicle_direction(self, bboxes):
        """
        차량의 이동 방향 감지 (이전 프레임과 비교)

        방법:
        - 이전 프레임의 BBOX 중심과 현재 프레임의 BBOX 중심을 비교
        - 이동 벡터를 계산하여 방향 결정
        - 정지 상태인 경우 'STATIONARY' 반환

        Args:
            bboxes: 현재 프레임의 바운딩 박스 리스트 [[x1, y1, x2, y2], ...]

        Returns:
            list: 각 BBOX에 대한 방향 정보
                  [{'direction': 'N', 'angle': 0, 'confidence': 0.95, 'speed': 5.2}, ...]
        """
        import math

        directions = []
        current_centers = []

        # 현재 BBOX의 중심 계산
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_centers.append((center_x, center_y))

        # 이전 프레임이 있으면 비교
        if self.prev_bboxes and len(self.prev_bboxes) > 0:
            for curr_center in current_centers:
                curr_x, curr_y = curr_center

                # 가장 가까운 이전 BBOX 찾기
                min_dist = float('inf')
                closest_prev = None

                for prev_center in self.prev_bboxes:
                    prev_x, prev_y = prev_center
                    dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_prev = (prev_x, prev_y)

                if closest_prev and min_dist < 100:  # 100픽셀 이내만 동일 차량으로 간주
                    prev_x, prev_y = closest_prev
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y

                    # 이동 거리 (속도)
                    speed = math.sqrt(dx**2 + dy**2)

                    # 정지 상태 판단 (이동 거리가 3픽셀 미만)
                    if speed < 3.0:
                        directions.append({
                            'direction': 'STATIONARY',
                            'angle': 0,
                            'confidence': 1.0,
                            'speed': speed
                        })
                    else:
                        # 각도 계산 (라디안 → 도)
                        # atan2(dy, dx)는 오른쪽(동쪽)을 0도로 하는 각도
                        angle = math.atan2(dy, dx) * 180 / math.pi

                        # 8방향 분류
                        direction = self._angle_to_direction(angle)

                        # 신뢰도: 이동 거리가 클수록 높음 (최대 1.0)
                        confidence = min(1.0, speed / 20.0)

                        directions.append({
                            'direction': direction,
                            'angle': angle,
                            'confidence': confidence,
                            'speed': speed
                        })
                else:
                    # 새로 등장한 차량
                    directions.append({
                        'direction': 'NEW',
                        'angle': 0,
                        'confidence': 0.5,
                        'speed': 0
                    })
        else:
            # 첫 프레임: 모든 차량이 새로 등장
            directions = [{'direction': 'NEW', 'angle': 0, 'confidence': 0.5, 'speed': 0}] * len(bboxes)

        # 현재 프레임을 이전 프레임으로 저장
        self.prev_bboxes = current_centers

        return directions
    
    @staticmethod
    def _angle_to_direction(angle):
        """
        각도를 8방향으로 변환

        Args:
            angle: 각도 (-180 ~ 180)

        Returns:
            str: 방향 (N, NE, E, SE, S, SW, W, NW)
        """
        # 정규화: 0 ~ 360
        angle = angle % 360

        # 8방향 매핑
        directions = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
        index = int((angle + 22.5) / 45) % 8

        return directions[index]

    @staticmethod
    def adjust_bbox_for_diagonal_movement(bbox, direction, shrink_ratio=0.25):
        """
        대각선 이동 차량의 BBOX를 중심부만 남기도록 축소

        대각선(NE, SE, SW, NW) 이동 시:
        - BBOX를 중심 기준으로 축소하여 차량의 정중앙만 사용
        - 기본 25% 크기로 축소 (75% 제거)하여 외곽 모서리 부분 제거

        Args:
            bbox: [x1, y1, x2, y2] 바운딩 박스
            direction: 이동 방향 ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 등)
            shrink_ratio: 유지할 비율 (0.25 = 25% 유지, 75% 제거)

        Returns:
            list: [x1_new, y1_new, x2_new, y2_new] 조정된 바운딩 박스
        """
        # 대각선 방향만 처리
        diagonal_directions = ['NE', 'SE', 'SW', 'NW']

        if direction not in diagonal_directions:
            # 대각선이 아니면 원본 반환
            return bbox

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 중심 기준으로 축소 (shrink_ratio 크기로)
        new_width = width * shrink_ratio
        new_height = height * shrink_ratio

        x1_new = int(center_x - new_width / 2)
        y1_new = int(center_y - new_height / 2)
        x2_new = int(center_x + new_width / 2)
        y2_new = int(center_y + new_height / 2)

        return [x1_new, y1_new, x2_new, y2_new]

