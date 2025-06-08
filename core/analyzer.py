import cv2, csv, datetime, time, os
import mediapipe as mp
from utils.config import POSTURE_THRESHOLD, LOG_FILE
from core.preprocessing import (
    correct_perspective,
    rotate_and_scale,
    histogram_equalization,
    denoise,
    sharpen
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ALARM_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)

class Analyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ensure_log_dir()
        self.ensure_log_header()
        self.enable_preprocessing = False  # 实时分析默认关闭预处理

    def ensure_log_dir(self):
        log_path = './logs'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def ensure_log_header(self):
        # 如果文件不存在或为空，则写入表头
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            try:
                with open(LOG_FILE, 'w', newline='') as f:
                    csv.writer(f).writerow([
                        'Timestamp',           # 时间戳
                        'EarShoulderDiff',     # 耳-肩垂直差值
                        'ShoulderHipDiff',     # 肩-髋垂直差值
                        'PostureStatus'        # 姿势状态（True/False）
                    ])
            except Exception as e:
                print(f"文件初始化失败: {str(e)}")

    def analyze(self, frame, src_points=None, dst_points=None, angle=0, scale=1.0, preprocessing=False):
        # 仅图片分析时开启预处理
        if preprocessing:
            if src_points is not None and dst_points is not None:
                frame = correct_perspective(frame, src_points, dst_points)
            frame = rotate_and_scale(frame, angle, scale)
            frame = histogram_equalization(frame)
            frame = denoise(frame)
            frame = sharpen(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        posture_status = False
        ear_shoulder_diff = 0.0
        shoulder_hip_diff = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            def get_landmark_avg(landmark1, landmark2):
                try:
                    return (landmark1.y + landmark2.y) / 2
                except:
                    return 0.0

            ear_y = get_landmark_avg(
                landmarks[mp_pose.PoseLandmark.LEFT_EAR],
                landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
            )
            shoulder_y = get_landmark_avg(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            )
            hip_y = get_landmark_avg(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            )

            ear_shoulder_diff = float(ear_y - shoulder_y)
            shoulder_hip_diff = float(shoulder_y - hip_y)
            posture_status = bool(
                (ear_shoulder_diff > POSTURE_THRESHOLD) and
                (abs(shoulder_hip_diff) < 0.62)
            )

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=ALARM_COLOR if posture_status else NORMAL_COLOR,
                    thickness=2
                )
            )

            if posture_status:
                cv2.rectangle(image, (0, 0),
                              (image.shape[1] - 1, image.shape[0] - 1),
                              ALARM_COLOR, 10)
                cv2.putText(image, "POSTURE WARNING!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ALARM_COLOR, 3)

        return image, posture_status, {
            'ear_shoulder': f"{ear_shoulder_diff:.4f}",
            'shoulder_hip': f"{shoulder_hip_diff:.4f}"
        }

    def log(self, status, metrics):
        self.ensure_log_dir()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, metrics['ear_shoulder'], metrics['shoulder_hip'], status])

    def run_realtime(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            try:
                image, posture_status, metrics = self.analyze(frame)
                self.log(posture_status, metrics)
                cv2.putText(image,
                            f"Ear-Shoulder: {metrics['ear_shoulder']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(image,
                            f"Shoulder-Hip: {metrics['shoulder_hip']}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(image,
                            f"Status: {'WARNING' if posture_status else 'NORMAL'}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            ALARM_COLOR if posture_status else NORMAL_COLOR, 2)
                cv2.imshow('Posture Correction', image)
            except Exception as e:
                print(f"处理异常: {str(e)}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()