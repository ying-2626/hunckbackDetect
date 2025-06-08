from flask_mail import Message
from utils.config import EMAIL_FROM, EMAIL_TO

class Emailer:
    def __init__(self, mail):
        self.mail = mail

    def send(self, report, to_address=EMAIL_TO):
        msg = Message(
            subject='Posture Anomaly Report',
            sender=EMAIL_FROM,
            recipients=[to_address] if isinstance(to_address, str) else to_address
        )
        msg.body = report
        self.mail.send(msg)
