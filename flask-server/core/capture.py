import threading
import time
import cv2


class CameraCapture:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, camera_index=0):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.camera_index = camera_index
            cls._instance.max_retry = 3
            cls._instance._init_camera()
        return cls._instance

    def _init_camera(self):
        for i in range(self.max_retry):
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return
            time.sleep(0.5)
        raise RuntimeError(f"无法初始化摄像头 index={self.camera_index}")

    def get_frame(self):
        with self._lock:
            for _ in range(self.max_retry):
                ret, frame = self.cap.read()
                if ret:
                    return frame
                # 自动恢复
                self._init_camera()
            raise RuntimeError("无法获取视频帧")

    def release(self):
        with self._lock:
            if self.cap.isOpened():
                self.cap.release()
                CameraCapture._instance = None
