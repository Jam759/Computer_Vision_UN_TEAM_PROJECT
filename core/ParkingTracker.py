from collections import defaultdict


class ParkingTracker:
    """주차된 자동차를 추적하는 클래스 (프레임 기반)"""
    
    def __init__(self, fps=30, parking_seconds=5, unparking_seconds=10):
        """
        Args:
            fps: 영상의 프레임 레이트
            parking_seconds: 주차로 인정하는 최소 시간 (초)
            unparking_seconds: 주차됨에서 비워짐으로 변경하는 최소 시간 (초)
        """
        self.fps = fps
        self.parking_frames = int(fps * parking_seconds)      # 30초 = fps*30 프레임
        self.unparking_frames = int(fps * unparking_seconds)  # 10초 = fps*10 프레임
        
        self.vehicle_tracking = {}    # {roi_id: frame_count}
        self.parked_vehicles = {}     # {roi_id: frame_count}
        self.leaving_vehicles = {}    # {roi_id: frame_count}
    
    def update(self, vehicles_in_roi):
        """
        각 프레임에서 호출하여 주차 상태 업데이트
        
        Args:
            vehicles_in_roi: {roi_id: bool} 형식으로 각 ROI에 자동차가 있는지 여부
        
        Returns:
            dict: 업데이트된 주차 상태 정보
        """
        newly_parked = set()
        newly_empty = set()
        
        # 현재 자동차가 있는 ROI 업데이트
        for roi_id, has_vehicle in vehicles_in_roi.items():
            if has_vehicle:
                # 자동차가 있는 경우
                if roi_id not in self.vehicle_tracking:
                    # 새로운 자동차 감지
                    self.vehicle_tracking[roi_id] = 1
                else:
                    self.vehicle_tracking[roi_id] += 1
                
                # 주차 조건 확인: fps*30 프레임(30초) 이상 지속
                if self.vehicle_tracking[roi_id] >= self.parking_frames and roi_id not in self.parked_vehicles:
                    self.parked_vehicles[roi_id] = 0
                    newly_parked.add(roi_id)
                    # 떠나는 상태 초기화
                    if roi_id in self.leaving_vehicles:
                        del self.leaving_vehicles[roi_id]
            else:
                # 자동차가 없는 경우
                if roi_id in self.vehicle_tracking:
                    del self.vehicle_tracking[roi_id]
                
                # 주차됨 상태에서 비워짐 상태로 전환
                if roi_id in self.parked_vehicles:
                    # 처음 떠난 것을 감지하면 카운트 시작
                    if roi_id not in self.leaving_vehicles:
                        self.leaving_vehicles[roi_id] = 0
                    else:
                        self.leaving_vehicles[roi_id] += 1
                        # fps*10 프레임(10초) 이상 자동차가 감지되지 않으면 비워짐 처리
                        if self.leaving_vehicles[roi_id] >= self.unparking_frames:
                            del self.parked_vehicles[roi_id]
                            del self.leaving_vehicles[roi_id]
                            newly_empty.add(roi_id)
        
        return {
            'parked_count': len(self.parked_vehicles),
            'tracked_count': len(self.vehicle_tracking),
            'newly_parked': newly_parked,
            'newly_empty': newly_empty
        }
    
    def get_parked_count(self):
        """주차된 자동차 수"""
        return len(self.parked_vehicles)
    
    def get_tracking_frames(self, roi_id):
        """특정 ROI에서 추적된 프레임 수 반환"""
        if roi_id in self.vehicle_tracking:
            return self.vehicle_tracking[roi_id]
        return 0
    
    def reset(self):
        """추적 상태 리셋"""
        self.vehicle_tracking = {}
        self.parked_vehicles = {}
        self.leaving_vehicles = {}
