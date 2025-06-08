from flask import Flask, jsonify, request, Response, render_template, render_template_string, send_file
from flask_mail import Mail
import cv2
import numpy as np
import os
import tempfile
import time
from flask_cors import CORS

from core.analyzer import Analyzer
from core.capture import CameraCapture
from core.scheduler import Scheduler
from utils.config import SERVER_CONFIG

app = Flask(__name__)
CORS(app)
mail = Mail(app)
s = Scheduler(mail, app)  # 传入app实例

analyzer_instance = Analyzer()  # 全局Analyzer实例，避免重复初始化


@app.route('/video_feed')
def video_feed():
    def generate():
        camera = CameraCapture()
        while True:
            frame = camera.get_frame()
            # 实时分析不做预处理
            image, posture_status, metrics = analyzer_instance.analyze(frame, preprocessing=False)
            ret, jpeg = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': "未上传图片"}), 400

    file = request.files['image']
    print(f"收到文件: {file.filename}")

    if file.filename == '':
        return jsonify({'error': "未选择图片"}), 400

    # 读取图片为OpenCV格式
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(f"图片读取成功: {img is not None}")

    if img is None:
        return jsonify({'error': "图片格式错误"}), 400

    # 进行预处理和姿势分析（不写入日志）
    image, posture_status, metrics = analyzer_instance.analyze(img, preprocessing=True)

    # 保存标注后图片到 static/tmp 目录
    tmp_dir = os.path.join(os.path.dirname(__file__), "static", "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    filename = f"analyzed_{int(time.time())}.jpg"
    out_path = os.path.join(tmp_dir, filename)
    cv2.imwrite(out_path, image)
    print(f"处理后图片保存路径: {out_path}")

    # 返回分析结果和图片下载链接
    return jsonify({
        'ear_shoulder': metrics['ear_shoulder'],
        'shoulder_hip': metrics['shoulder_hip'],
        'posture_status': '异常' if posture_status else '正常',
        'filename': filename
    })


@app.route('/download_analyzed')
def download_analyzed():
    filename = request.args.get('filename')
    tmp_dir = os.path.join(os.path.dirname(__file__), "logs", "tmp")
    file_path = os.path.join(tmp_dir, filename) if filename else None

    if not file_path or not os.path.exists(file_path):
        return "文件不存在", 404
    return send_file(file_path, mimetype='image/jpeg')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})


def start_background_tasks():
    s.start()

if __name__ == '__main__':
    start_background_tasks()
    app.run(host='0.0.0.0', port=SERVER_CONFIG['PORT'])
