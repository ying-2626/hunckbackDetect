import smtplib
from email.mime.text import MIMEText
from utils.config import EMAIL_FROM, EMAIL_PASSWORD


class Emailer:
    def __init__(self, mail, app):
        self.mail = mail
        self.app = app

    def send(self, report, to_address):
        try:
            msg = MIMEText(report)
            msg['Subject'] = 'Posture Anomaly Report'
            msg['From'] = EMAIL_FROM
            msg['To'] = to_address

            # 创建连接但不使用上下文管理器
            server = smtplib.SMTP_SSL('smtp.qq.com', 465)
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
            # 尝试正常退出
            try:
                server.quit()  # 发送QUIT命令
            except:
                pass  # 忽略退出错误
            return True

        except Exception as e:
            print(f"邮件发送失败: {str(e)}")
            return False