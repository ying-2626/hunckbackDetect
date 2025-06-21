import cv2, csv, datetime, os, math, time
import mediapipe as mp
import threading
import numpy as np
# from playsound import playsound  # 移除声音报警
from utils.config import POSTURE_THRESHOLD, LOG_FILE
from core.preprocessing import (
    correct_perspective,
    rotate_and_scale,
    denoise,
    sharpen,
    rotate_scale_translate,
    adaptive_histogram_equalization
)
from core.detecter import (
    detect_person_and_fit_rect,
    compute_preproc_params_alternative,
    find_upright_rotation
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ALARM_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)
HUNCHBACK_ANGLE_THRESHOLD = 30
HUNCHBACK_ALARM_COLOR = (0, 0, 255)  # 红色警报


class Analyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.ensure_log_dir()
        self.ensure_log_header()
        self.enable_preprocessing = False

        # self.last_alarm_time = 0
        # self.alarm_cooldown = 3  # 警报冷却时间（秒）
        # self.alert_lock = threading.Lock()  # 新增警报锁
        # self.active_alerts = set()  # 跟踪当前活动的警报类型

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
                        'Timestamp',
                        'EarShoulderDiff',
                        'ShoulderHipDiff',
                        'PostureStatus',
                        'SpineAngle',
                        'HunchbackStatus'
                    ])
            except Exception as e:
                print(f"文件初始化失败: {str(e)}")

    def midpoint(self, point1, point2):
        x = (point1.x + point2.x) / 2
        y = (point1.y + point2.y) / 2
        return (x, y)

    def calculate_spine_angle(self, landmarks):
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            shoulder_mid = self.midpoint(left_shoulder, right_shoulder)
            hip_mid = self.midpoint(left_hip, right_hip)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))
            return angle if dy > 0 else 0
        except:
            return 0

    # def trigger_alert(self, alert_type="hunchback"):
    #     """非阻塞方式触发警报"""
    #     current_time = time.time()

    #     # 使用锁确保线程安全
    #     with self.alert_lock:
    #         # 检查冷却时间和活动警报
    #         if (current_time - self.last_alarm_time < self.alarm_cooldown or
    #                 alert_type in self.active_alerts):
    #             return

    #         # 标记活动���报
    #         self.active_alerts.add(alert_type)
    #         self.last_alarm_time = current_time

    #     # 在后台线程中播放声音
    #     def play_sound():
    #         try:
    #             sound_file = "hunchback_alert.wav" if alert_type == "hunchback" else "posture_alert.wav"
    #             playsound(sound_file)
    #         except Exception as e:
    #             print(f"声音报警失败: {e}")
    #         finally:
    #             # 播放完成后移除活动警报标记
    #             with self.alert_lock:
    #                 self.active_alerts.discard(alert_type)

    #     threading.Thread(target=play_sound, daemon=True).start()

    def analyze(self, frame, src_points=None, dst_points=None, preprocessing=False):
        if preprocessing:
            """
            # 1. 用mediapipe检测一次，获取关键点
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            upright_needed = False
            upright_angle = 0
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # 取左右肩膀和左右耳朵
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                h, w = frame.shape[:2]
                # 计算肩膀连线与水平线夹角
                dx = (r_shoulder.x - l_shoulder.x) * w
                dy = (r_shoulder.y - l_shoulder.y) * h
                angle = abs(math.degrees(math.atan2(dy, dx)))
                # 计算耳朵和肩膀的y坐标
                avg_ear_y = (l_ear.y + r_ear.y) / 2
                avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                # 判断是否本来正立
                if angle > 45 and avg_ear_y >= avg_shoulder_y:
                    upright_needed = False

            # 2. 若需要正立，自动旋转
            if upright_needed:
                upright_angle = find_upright_rotation(frame, self.pose)
                if upright_angle != 0:
                    frame = rotate_and_scale(frame, -upright_angle, 1.0)
"""
            # 3. 透视校正
            if src_points is not None and dst_points is not None:
                frame = correct_perspective(frame, src_points, dst_points)

            # 5. CLAHE 增强
            frame = adaptive_histogram_equalization(frame)
            # 6. 去噪、锐化
            frame = denoise(frame)
            frame = sharpen(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        posture_status = False
        hunchback_status = False
        ear_shoulder_diff = 0.0
        shoulder_hip_diff = 0.0
        metrics = {}

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
                (abs(shoulder_hip_diff) < 0.72)
            )

            # 脊柱角度与驼背检测
            spine_angle = self.calculate_spine_angle(landmarks)
            hunchback_status = self.detect_hunchback(spine_angle)
            metrics['spine_angle'] = f"{spine_angle:.1f}°"

            # 用mediapipe自带的标注
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=ALARM_COLOR if posture_status or hunchback_status else NORMAL_COLOR,
                    thickness=2
                )
            )
            if hunchback_status:
                cv2.putText(image, "HUNCHBACK WARNING!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, HUNCHBACK_ALARM_COLOR, 3)
            if posture_status:
                cv2.rectangle(image, (0, 0),
                              (image.shape[1] - 1, image.shape[0] - 1),
                              ALARM_COLOR, 10)
                cv2.putText(image, "POSTURE WARNING!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ALARM_COLOR, 3)
        metrics.update({
            'ear_shoulder': f"{ear_shoulder_diff:.4f}",
            'shoulder_hip': f"{shoulder_hip_diff:.4f}"
        })

        return image, posture_status, hunchback_status, metrics

    def log(self, status, metrics, hunchback_status=0, spine_angle=""):
        self.ensure_log_dir()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                timestamp,
                metrics.get('ear_shoulder', ''),
                metrics.get('shoulder_hip', ''),
                status,
                spine_angle,
                hunchback_status
            ])

    def run_realtime(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            try:
                image, posture_status, hunchback_status, metrics = self.analyze(frame)
                self.log(posture_status, metrics, hunchback_status, metrics.get('spine_angle', ''))
                cv2.putText(image, f"Spine Angle: {metrics.get('spine_angle', 'N/A')}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            HUNCHBACK_ALARM_COLOR if hunchback_status else (0, 0, 0),
                            2)
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
                cv2.putText(image,
                            f"Status: {'HUNCHBACK' if hunchback_status else ('WARNING' if posture_status else 'NORMAL')}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            HUNCHBACK_ALARM_COLOR if hunchback_status else (
                                ALARM_COLOR if posture_status else NORMAL_COLOR), 2)
                cv2.imshow('Posture Monitor', image)
            except Exception as e:
                print(f"处理异常: {str(e)}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_hunchback(self, spine_angle):
        """检测驼背状态"""
        return spine_angle > HUNCHBACK_ANGLE_THRESHOLD
