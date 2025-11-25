"""
주차 상태 추적 모듈 (상태 머신 기반)
"""


class ParkingTracker:
    """
    주차 상태를 상태 머신으로 관리하는 클래스
    
    상태 전이:
    EMPTY -> DETECTING (차량 감지) -> PARKED (주차 확정)
    PARKED -> LEAVING (차량 미감지) -> EMPTY (비워짐 확정)
    """
    
    # 상태 상수
    EMPTY = 0        # 비어있음
    DETECTING = 1    # 차량 감지 중 (주차 판정 전)
    PARKED = 2       # 주차됨
    LEAVING = 3      # 떠나는 중 (비워짐 판정 전)
    
    # 상태명 매핑 (디버깅용)
    STATE_NAMES = {
        0: "EMPTY",
        1: "DETECTING",
        2: "PARKED",
        3: "LEAVING"
    }
    
    def __init__(self, fps=30, parking_seconds=3, unparking_seconds=2, detection_tolerance_seconds=1):
        """
        Args:
            fps: 영상 프레임율
            parking_seconds: 주차 확정 시간 (초)
            unparking_seconds: 비워짐 확정 시간 (초)
            detection_tolerance_seconds: YOLO 감지 실패 허용 시간 (초)
        """
        self.fps = fps
        self.parking_frames = int(fps * parking_seconds)                    # 주차 판정 프레임
        self.unparking_frames = int(fps * unparking_seconds)                # 비워짐 판정 프레임
        self.tolerance_frames = int(fps * detection_tolerance_seconds)      # 감지 실패 허용 프레임 (1초)
        
        # ROI별 상태: {roi_id: {'state': int, 'frame_count': int, 'miss_count': int}}
        self.roi_states = {}
    
    def update(self, vehicles_in_roi):
        """
        각 프레임에서 호출하여 주차 상태 업데이트
        
        Args:
            vehicles_in_roi: {roi_id: bool} 각 ROI에 차량 여부
        
        Returns:
            dict: {
                'parked_count': int,          # 주차된 ROI 개수
                'tracked_count': int,         # 감지/주차된 ROI 개수
                'newly_parked': set,          # 새로 주차된 ROI
                'newly_empty': set            # 새로 비워진 ROI
            }
        """
        newly_parked = set()
        newly_empty = set()
        
        for roi_id, has_vehicle in vehicles_in_roi.items():
            # ROI 상태 초기화
            if roi_id not in self.roi_states:
                self.roi_states[roi_id] = {'state': self.EMPTY, 'frame_count': 0, 'miss_count': 0}
            
            state_info = self.roi_states[roi_id]
            current_state = state_info['state']
            
            if has_vehicle:
                self._handle_vehicle_detected(roi_id, state_info, newly_parked)
            else:
                self._handle_vehicle_not_detected(roi_id, state_info, newly_empty)
        
        return {
            'parked_count': self._count_state(self.PARKED),
            'tracked_count': self._count_state(self.DETECTING) + self._count_state(self.PARKED),
            'newly_parked': newly_parked,
            'newly_empty': newly_empty
        }
    
    def _handle_vehicle_detected(self, roi_id, state_info, newly_parked):
        """차량 감지 시 상태 처리"""
        current_state = state_info['state']
        
        # 감지 실패 카운트 리셋 (다시 감지됨)
        state_info['miss_count'] = 0
        
        if current_state == self.EMPTY:
            # 빈 상태 -> 감지 상태
            state_info['state'] = self.DETECTING
            state_info['frame_count'] = 1
        
        elif current_state == self.DETECTING:
            # 감지 상태 유지
            state_info['frame_count'] += 1
            
            # 주차 확정 조건
            if state_info['frame_count'] >= self.parking_frames:
                state_info['state'] = self.PARKED
                state_info['frame_count'] = 0
                newly_parked.add(roi_id)
        
        elif current_state == self.PARKED:
            # 주차됨 상태 유지
            state_info['frame_count'] = 0
        
        elif current_state == self.LEAVING:
            # 떠나는 상태 -> 주차됨 상태 복귀
            state_info['state'] = self.PARKED
            state_info['frame_count'] = 0
    
    def _handle_vehicle_not_detected(self, roi_id, state_info, newly_empty):
        """차량 미감지 시 상태 처리 (1초 허용)"""
        current_state = state_info['state']
        
        if current_state == self.EMPTY:
            # 이미 빈 상태
            pass
        
        elif current_state == self.DETECTING:
            # 감지 중 미감지: 감지 실패 허용 (1초)
            state_info['miss_count'] += 1
            
            if state_info['miss_count'] >= self.tolerance_frames:
                # 1초 이상 미감지 -> 빈 상태로
                state_info['state'] = self.EMPTY
                state_info['frame_count'] = 0
                state_info['miss_count'] = 0
        
        elif current_state == self.PARKED:
            # 주차됨 상태에서 미감지: 감지 실패 허용 (1초)
            state_info['miss_count'] += 1
            
            if state_info['miss_count'] >= self.tolerance_frames:
                # 1초 이상 미감지 -> 떠나는 상태로
                state_info['state'] = self.LEAVING
                state_info['frame_count'] = 1
                state_info['miss_count'] = 0
        
        elif current_state == self.LEAVING:
            # 떠나는 상태 유지
            state_info['frame_count'] += 1
            
            # 비워짐 확정 조건
            if state_info['frame_count'] >= self.unparking_frames:
                state_info['state'] = self.EMPTY
                state_info['frame_count'] = 0
                state_info['miss_count'] = 0
                newly_empty.add(roi_id)
    
    def _count_state(self, state):
        """특정 상태의 ROI 개수"""
        return sum(1 for info in self.roi_states.values() if info['state'] == state)
    
    def get_roi_state(self, roi_id):
        """특정 ROI의 현재 상태 반환"""
        if roi_id in self.roi_states:
            return self.roi_states[roi_id]['state']
        return self.EMPTY
    
    def get_parked_count(self):
        """주차된 ROI 개수"""
        return self._count_state(self.PARKED)
    
    def get_state_name(self, roi_id):
        """ROI의 상태명 반환 (디버깅용)"""
        state = self.get_roi_state(roi_id)
        return self.STATE_NAMES.get(state, "UNKNOWN")
    
    def reset(self):
        """전체 상태 리셋"""
        self.roi_states = {}
