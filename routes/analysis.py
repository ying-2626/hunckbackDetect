from flask import Blueprint, request, jsonify, Response, send_file, current_app
import cv2
import numpy as np
import base64
import time
import os

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/video_feed')
def video_feed():
    # Access globals from current_app config or extension
    # We will assume they are stored in current_app.extensions['hunchback']
    # or just accessible via a shared module if we implemented one.
    # For Blueprint pattern, it's better to use current_app.
    
    app_context = current_app.extensions.get('hunchback', {})
    camera_instance = app_context.get('camera_instance')
    analyzer_instance = app_context.get('analyzer_instance')
    is_vercel = app_context.get('is_vercel', False)

    if is_vercel or camera_instance is None:
        return "Real-time video feed not available in serverless environment.", 503
        
    def generate():
        while True:
            frame = camera_instance.get_frame()
            # 实时分析不做预处理
            image, posture_status, hunchback_status, metrics = analyzer_instance.analyze(frame, preprocessing=False)
            ret, jpeg = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@analysis_bp.route('/analyze', methods=['POST'])
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

    app_context = current_app.extensions.get('hunchback', {})
    analyzer_instance = app_context.get('analyzer_instance')
    is_vercel = app_context.get('is_vercel', False)

    # 进行预处理和姿势分析（不写入日志）
    image, posture_status, hunchback_status, metrics = analyzer_instance.analyze(img, preprocessing=True)

    # 总是返回 Base64，避免文件写入问题，特别是 Vercel 环境
    retval, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 为了兼容，如果非 Vercel 环境，还是尝试写入文件（可选，但推荐只用 Base64）
    filename = f"analyzed_{int(time.time())}.jpg"
    
    if not is_vercel:
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            tmp_dir = os.path.join(BASE_DIR, "static", "tmp")
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            out_path = os.path.join(tmp_dir, filename)
            cv2.imwrite(out_path, image)
            print(f"处理后图片保存路径: {out_path}")
        except Exception as e:
            print(f"Warning: Failed to save image to disk: {e}")

    # 返回分析结果和图片下载链接/Base64
    return jsonify({
        'ear_shoulder': metrics['ear_shoulder'],
        'shoulder_hip': metrics['shoulder_hip'],
        'posture_status': '异常' if posture_status else '正常',
        'hunchback_status': '异常' if hunchback_status else '正常',
        'spine_angle': metrics.get('spine_angle', ''),
        'filename': filename,
        'image_base64': image_base64
    })


@analysis_bp.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': "未上传图片"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': "未选择图片"}), 400
    try:
        # 读取图片为字节流
        image_bytes = file.read()
        
        app_context = current_app.extensions.get('hunchback', {})
        classifier_instance = app_context.get('classifier_instance')
        
        # 调用分类方法
        result = classifier_instance.classify_image(image_bytes)
        # 添加调试输出
        print(f"分类结果: {result}")

        # 确保返回结果包含姿态和置信度
        if 'class' in result and 'conf' in result:
            response_data = {
                'posture': result['class'],  # good 或 bad
                'confidence': float(result['conf'])  # 置信度
            }
            # 透传详细分析数据
            if 'analysis' in result:
                response_data['analysis'] = result['analysis']
            
            return jsonify(response_data)
        else:
            return jsonify({'error': "无效的分类结果"}), 500

    except Exception as e:
        print(f"处理分类请求时出错: {str(e)}")
        return jsonify({'error': f"处理请求时发生错误: {str(e)}"}), 500


@analysis_bp.route('/download_analyzed')
def download_analyzed():
    filename = request.args.get('filename')
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    tmp_dir = os.path.join(BASE_DIR, "logs", "tmp")
    file_path = os.path.join(tmp_dir, filename) if filename else None

    if not file_path or not os.path.exists(file_path):
        return "文件不存在", 404
    return send_file(file_path, mimetype='image/jpeg')
