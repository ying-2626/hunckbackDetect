from flask import Flask, request, jsonify
from inference import YOLOService, RequestHandler
from rag import RAGService  # 引入 RAG 服务

# Flask应用
app = Flask(__name__)

# 初始化YOLO服务和请求处理器
model_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\YOLOv8\runs\detect\train3\weights\best.pt"
# model_path = "/home/whs/PoseDetection/SittingWatch/YOLOv8/runs/detect/train3/weights/best.pt"
yolo_service = YOLOService(model_path)
request_handler = RequestHandler(yolo_service)
rag_service = RAGService() # 初始化 RAG 服务

@app.route('/detect', methods=['POST'])
def detect():
    """处理图片检测请求"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if not image_file.filename.endswith('.jpg'):
        return jsonify({"error": "Only JPG images are supported"}), 400
    
    return request_handler.handle_request(image_file)

@app.route('/report', methods=['GET'])
def get_report():
    """生成周报接口"""
    try:
        days = int(request.args.get('days', 7))
        report = rag_service.generate_weekly_report(days=days)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    """添加知识库接口"""
    data = request.json
    content = data.get('content')
    category = data.get('category', 'general')
    if not content:
        return jsonify({"error": "Content required"}), 400
    
    rag_service.add_knowledge(content, category)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # 使用Gunicorn运行时不需要app.run()
    pass
