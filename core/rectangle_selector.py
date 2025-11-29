"""
마우스로 ROI(주차칸) 선택 모듈
"""

import cv2
from core.utils import GeometryUtils


class RectangleSelector:
    """
    마우스 클릭으로 직사각형 주차칸을 선택하는 클래스
    
    - 클릭 4번으로 사각형 생성
    - 원본 좌표 저장 + 리사이징된 화면에 표시
    """
    
    def __init__(self, frame):
        """
        Args:
            frame: 입력 이미지 (numpy array)
        """
        self.original_frame = frame.copy()
        self.frame, self.scale_x, self.scale_y = GeometryUtils.resize_with_ratio(frame)
        self.temp_frame = self.frame.copy()
        
        self.points = []           # 현재 클릭 좌표
        self.rectangles = []       # 저장된 사각형 목록: [[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
        
        cv2.namedWindow("Frame")
        cv2.imshow("Frame", self.temp_frame)
        cv2.setMouseCallback("Frame", self.click_event)
    
    def run(self):
        """윈도우 실행 (ESC로 종료)"""
        while True:
            cv2.imshow("Frame", self.temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        self.close()
    
    def close(self):
        """윈도우 종료"""
        cv2.destroyAllWindows()
    
    def click_event(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # 마우스 좌표 -> 원본 영상 좌표 변환
        orig_x = int(x * self.scale_x)
        orig_y = int(y * self.scale_y)
        
        self._handle_add(orig_x, orig_y, x, y)
        cv2.imshow("Frame", self.temp_frame)
    
    def _handle_add(self, orig_x, orig_y, disp_x, disp_y):
        """주차칸 추가 처리"""
        # 원본 좌표로 저장
        self.points.append((orig_x, orig_y))
        
        # 리사이징된 좌표로 표시
        cv2.circle(self.temp_frame, (disp_x, disp_y), 5, (0, 255, 0), -1)
        
        if len(self.points) == 4:
            # 사각형 저장 및 그리기
            self.rectangles.append(self.points.copy())
            for i in range(4):
                p1_orig = self.points[i]
                p2_orig = self.points[(i + 1) % 4]
                p1_disp = (int(p1_orig[0] / self.scale_x), int(p1_orig[1] / self.scale_y))
                p2_disp = (int(p2_orig[0] / self.scale_x), int(p2_orig[1] / self.scale_y))
                cv2.line(self.temp_frame, p1_disp, p2_disp, (255, 0, 0), 2)
            self.points = []
    
    def _redraw(self):
        """현재 상태로 화면 다시 그리기"""
        self.temp_frame = self.frame.copy()
        
        for rect in self.rectangles:
            for i in range(4):
                p1_orig = rect[i]
                p2_orig = rect[(i + 1) % 4]
                p1_disp = (int(p1_orig[0] / self.scale_x), int(p1_orig[1] / self.scale_y))
                p2_disp = (int(p2_orig[0] / self.scale_x), int(p2_orig[1] / self.scale_y))
                cv2.line(self.temp_frame, p1_disp, p2_disp, (255, 0, 0), 2)