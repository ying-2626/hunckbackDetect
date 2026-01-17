from flask import Flask, request, jsonify
from inference import YOLOService, RequestHandler

# Flask应用
app = Flask(__name__)

# 初始化YOLO服务和请求处理器
model_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\YOLOv8\runs\detect\train3\weights\best.pt"
# model_path = "/home/whs/PoseDetection/SittingWatch/YOLOv8/runs/detect/train3/weights/best.pt"
yolo_service = YOLOService(model_path)
request_handler = RequestHandler(yolo_service)

@app.route('/detect', methods=['POST'])
def detect():
    """处理图片检测请求"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if not image_file.filename.endswith('.jpg'):
        return jsonify({"error": "Only JPG images are supported"}), 400
    
    return request_handler.handle_request(image_file)

if __name__ == '__main__':
    # 使用Gunicorn运行时不需要app.run()
    pass
