import tkinter as tk
import traceback
from app.parking_app import ParkingApp

def show_error(root, error_msg):
    """에러 메시지 팝업 표시"""
    tk.messagebox.showerror("오류", f"시스템 오류가 발생했습니다:\n{error_msg}")

if __name__ == "__main__":
    root = None
    app = None
    
    try:
        root = tk.Tk()
        app = ParkingApp(root)
        
        # 종료 시 정리
        def on_closing():
            try:
                if app:
                    app.cleanup()
            except:
                pass
            finally:
                if root:
                    root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    
    except Exception as e:
        # 심각한 에러 로깅
        error_msg = traceback.format_exc()
        print(f"[FATAL ERROR]\n{error_msg}")
        
        # 에러 팝업 시도
        try:
            from tkinter import messagebox
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("치명적 오류", 
                f"애플리케이션 실행 중 오류가 발생했습니다:\n{str(e)}")
            root_err.destroy()
        except:
            pass
