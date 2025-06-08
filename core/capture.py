import cv2

class CameraCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera initialization failed")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera capture failed")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
