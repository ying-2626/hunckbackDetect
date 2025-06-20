import time, threading
from core.capture import CameraCapture
from core.analyzer import Analyzer
from models.report_generator import ReportGenerator
from utils.emailer import Emailer
from utils.config import SAMPLING_INTERVAL, ANOMALY_WINDOW, ANOMALY_THRESHOLD, EMAIL_TO

class Scheduler:
    def __init__(self, mail, app):
        self.analyzer = Analyzer()
        self.reporter = ReportGenerator()
        self.emailer = Emailer(mail,app)
        self.anomaly_count = 0
        self.camera = CameraCapture()  # 持久化摄像头对象
        self.app = app  # 保存Flask app实例

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        try:
            while True:
                frame = self.camera.get_frame()
                _, status, _, metrics = self.analyzer.analyze(frame)
                self.analyzer.log(status, metrics)

                if status:
                    self.anomaly_count += 1
                else:
                    self.anomaly_count = 0

                if self.anomaly_count >= ANOMALY_THRESHOLD:
                    report = self.reporter.generate_report()
                    # 在Flask上下文中发送邮件
                    with self.app.app_context():
                        self.emailer.send(report, EMAIL_TO)
                    self.anomaly_count = 0
                time.sleep(SAMPLING_INTERVAL)
        finally:
            self.camera.release()
