SERVER_CONFIG = {'PORT': 5000}

POSTURE_THRESHOLD = -0.255
LOG_FILE = './logs/posture_log.csv'
SAMPLING_INTERVAL = 3       # seconds between frames
ANOMALY_WINDOW = 6          # seconds window
ANOMALY_THRESHOLD = ANOMALY_WINDOW // SAMPLING_INTERVAL

EMAIL_FROM = '1744772180@qq.com'
EMAIL_PASSWORD = 'urjlznsxfcxtbchi'
EMAIL_TO = '2528691862@qq.com'
MAIL_SERVER = 'smtp.qq.com'
MAIL_PORT = 465
MAIL_USE_SSL = True # 使用SSL