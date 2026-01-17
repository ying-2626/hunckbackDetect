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

    def save_alert_history(self, report):
        """保存异常报告到历史文件"""
        history_file = os.path.join(os.path.dirname(__file__), "..", "logs", "alert_history.json")
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except:
                    pass
            
            new_alert = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content": report,
                "type": "anomaly_alert"
            }
            
            history.append(new_alert)
            # 保留最近100条
            if len(history) > 100:
                history = history[-100:]
                
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存异常历史失败: {e}")

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

                    # 保存异常记录到历史文件
                    self.save_alert_history(report)

                    if email_to:
                        with self.app.app_context():
                            self.emailer.send(report, email_to)
                    else:
                        print("未找到可用的接收邮箱，邮件未发送。")
                    self.anomaly_count = 0
                time.sleep(SAMPLING_INTERVAL)
        finally:
            self.camera.release()
