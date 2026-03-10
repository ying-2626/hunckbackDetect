SERVER_CONFIG = {'PORT': 5000}

POSTURE_THRESHOLD = -0.255
LOG_FILE = './logs/posture_log.csv'
SAMPLING_INTERVAL = 3       # seconds between frames
ANOMALY_WINDOW = 10         # seconds window
ANOMALY_THRESHOLD = ANOMALY_WINDOW // SAMPLING_INTERVAL

EMAIL_FROM = 'SittingWatch@qq.com'
EMAIL_PASSWORD = 'urjlznsxfcxtbchi'
EMAIL_TO = ''  # 动态指定
MAIL_SERVER = 'smtp.qq.com'
MAIL_PORT = 465
MAIL_USE_SSL = True # 使用SSL

# 端云协同配置
CLOUD_ENABLED = False       # 云端功能开关，默认关闭（端侧离线保底）
RAG_ENABLED = True          # RAG 功能开关，默认开启
RETRIEVAL_STRATEGY = 'local_bm25'  # 检索策略: local_bm25 或 cloud_vector

# 数据库配置
DB_PATH = './hunchback_encrypted.db'
DB_ENCRYPT_KEY = None  # 从用户密码派生，实际使用时设置

# 画像与记忆配置
SHORT_TERM_MEMORY_EXPIRE_HOURS = 24
LONG_TERM_MEMORY_DAYS = 30
PROFILE_UPDATE_INTERVAL_DAYS = 7