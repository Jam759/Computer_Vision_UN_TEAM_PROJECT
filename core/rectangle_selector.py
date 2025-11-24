import cv2

class RectangleSelector:
    def __init__(self, frame, mode="add"):
        """
        frame : OpenCV 이미지 (numpy array)
        mode  : "add" 또는 "delete" (기본값 "add")
        """
        self.original_frame = frame.copy()
        self.frame = self.resize_frame(frame.copy())
        self.temp_frame = self.frame.copy()
        self.points = []
        self.rectangles = []      # [[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
        self.mode = mode          # 외부 인자로 설정 가능 (기본 "add")
        
        # 스케일 팩터 (원본과 표시된 이미지의 비율)
        h_orig, w_orig = self.original_frame.shape[:2]
        h_resized, w_resized = self.frame.shape[:2]
        self.scale_x = w_orig / w_resized
        self.scale_y = h_orig / h_resized

        cv2.namedWindow("Frame")
        cv2.imshow("Frame", self.temp_frame)
        cv2.setMouseCallback("Frame", self.click_event)
    
    def resize_frame(self, frame):
        """
        프레임을 화면 크기에 맞게 리사이징
        최대 1200x700 크기로 제한
        """
        h, w = frame.shape[:2]
        max_w, max_h = 1200, 700
        
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            return cv2.resize(frame, (new_w, new_h))
        
        return frame

    def set_mode(self, mode):
        """
            외부에서 모드를 바꿀 때 사용: 'add' 또는 'delete'
        """
        if mode not in ("add", "delete"):
            raise ValueError("mode must be 'add' or 'delete'")
        self.mode = mode
        print(f"모드 변경 → {self.mode}")

    def run(self):
        
        """
            윈도우를 띄우고 ESC를 눌러 종료할 때까지 루프를 돈다.
            내부적으로는 모드 변경을 하지 않으므로 외부에서 set_mode 호출로 제어할 것.
        """
        
        while True:
            cv2.imshow("Frame", self.temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        self.close()

    def close(self):
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 마우스 좌표를 원본 영상 좌표로 변환
        orig_x = int(x * self.scale_x)
        orig_y = int(y * self.scale_y)

        if self.mode == "add":
            self.handle_add(orig_x, orig_y, x, y)
        elif self.mode == "delete":
            self.handle_delete(orig_x, orig_y)

        cv2.imshow("Frame", self.temp_frame)

    # ADD MODE - 리사이징된 화면에 표시하되, 원본 좌표로 저장
    def handle_add(self, orig_x, orig_y, disp_x, disp_y):
        # 원본 영상 좌표로 저장
        self.points.append((orig_x, orig_y))
        
        # 리사이징된 좌표로 화면에 표시
        cv2.circle(self.temp_frame, (disp_x, disp_y), 5, (0, 255, 0), -1)

        if len(self.points) == 4:
            # 사각형 저장 (원본 좌표)
            self.rectangles.append(self.points.copy())
            # 그림 그리기 (리사이징된 좌표)
            for i in range(4):
                p1_disp = (int(self.points[i][0] / self.scale_x), int(self.points[i][1] / self.scale_y))
                p2_disp = (int(self.points[(i+1)%4][0] / self.scale_x), int(self.points[(i+1)%4][1] / self.scale_y))
                cv2.line(self.temp_frame, p1_disp, p2_disp, (255, 0, 0), 2)
            self.points = []

    # -----------------------------
    # DELETE MODE
    # -----------------------------
    def handle_delete(self, x, y):
        delete_index = None

        # 클릭한 점이 어떤 사각형 내부인지 검사
        for idx, rect in enumerate(self.rectangles):
            if self.point_in_polygon((x, y), rect):
                delete_index = idx
                break

        if delete_index is not None:
            print("사각형 삭제됨:", delete_index)
            self.rectangles.pop(delete_index)
            self.redraw()

    # -----------------------------
    # Point in polygon (4점 다각형)
    # -----------------------------
    def point_in_polygon(self, point, polygon):
        x, y = point
        cnt = 0

        for i in range(4):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1) % 4]
            # 수평 교차 검사 (ray casting)
            if (y1 > y) != (y2 > y):
                atX = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
                if x < atX:
                    cnt += 1

        return (cnt % 2) == 1

    # -----------------------------
    # 전체 다시 그림
    # -----------------------------
    def redraw(self):
        self.temp_frame = self.frame.copy()
        for rect in self.rectangles:
            for i in range(4):
                p1_disp = (int(rect[i][0] / self.scale_x), int(rect[i][1] / self.scale_y))
                p2_disp = (int(rect[(i+1)%4][0] / self.scale_x), int(rect[(i+1)%4][1] / self.scale_y))
                cv2.line(self.temp_frame, p1_disp, p2_disp, (255, 0, 0), 2)