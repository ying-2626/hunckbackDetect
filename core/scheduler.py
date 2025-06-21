import time, threading


from core.analyzer import Analyzer
from models.report_generator import ReportGenerator
from utils.emailer import Emailer
from utils.config import SAMPLING_INTERVAL, ANOMALY_WINDOW, ANOMALY_THRESHOLD
import os
import json

class Scheduler:
    def __init__(self, mail, app,camera):
        self.analyzer = Analyzer()
        self.reporter = ReportGenerator()
        self.emailer = Emailer(mail, app)
        self.anomaly_count = 0
        self.camera = camera  # 使用传入的摄像头实例
        self.app = app  # 保存Flask app实例

    def get_latest_user_email(self):
        users_file = os.path.join(os.path.dirname(__file__), "..", "users.json")
        if not os.path.exists(users_file):
            return ""
        with open(users_file, "r", encoding="utf-8") as f:
            users = json.load(f)
            if users:
                # 取最后一个注册的用户邮箱
                return list(users.keys())[-1]
        return ""

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        try:
            while True:
                try:
                    frame = self.camera.get_frame()
                except Exception as e:
                    print(f"[Scheduler] 摄像头读取失败: {e}")
                    time.sleep(1)
                    continue  # 跳过本次循环，重试

                _, status, _, metrics = self.analyzer.analyze(frame)
                self.analyzer.log(status, metrics)

                if status:
                    self.anomaly_count += 1
                else:
                    self.anomaly_count = 0

                if self.anomaly_count >= ANOMALY_THRESHOLD:
                    report = self.reporter.generate_report()
                    # 动态获取最新用户邮箱
                    email_to = self.get_latest_user_email()
                    if email_to:
                        with self.app.app_context():
                            self.emailer.send(report, email_to)
                    else:
                        print("未找到可用的接收邮箱，邮件未发送。")
                    self.anomaly_count = 0
                time.sleep(SAMPLING_INTERVAL)
        finally:
            self.camera.release()
