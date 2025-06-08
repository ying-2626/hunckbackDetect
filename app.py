from flask import Flask, jsonify, request, Response
from flask_mail import Mail
import cv2

from core.analyzer import Analyzer
from core.capture import CameraCapture
from core.scheduler import Scheduler
from utils.config import SERVER_CONFIG

app = Flask(__name__)
mail = Mail(app)
s = Scheduler(mail, app)  # 传入app实例

analyzer_instance = Analyzer()  # 全局Analyzer实例，避免重复初始化


@app.route('/video_feed')
def video_feed():
    def generate():
        camera = CameraCapture()
        while True:
            frame = camera.get_frame()
            # 分析并绘制关键点和状态
            image, posture_status, metrics = analyzer_instance.analyze(frame)
            # 编码为JPEG
            ret, jpeg = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

def start_background_tasks():
    s.start()

if __name__ == '__main__':
    start_background_tasks()
    app.run(host='0.0.0.0', port=SERVER_CONFIG['PORT'])
