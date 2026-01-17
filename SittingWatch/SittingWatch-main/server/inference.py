import os
import queue
import threading
import json
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
from flask import jsonify

# YOLO服务类，封装模型加载和推理逻辑
class YOLOService:
    def __init__(self, model_path):
        """初始化YOLO模型"""
        self.model = YOLO(model_path)

    def infer(self, image_path):
        """对图片进行推理，返回分类和置信度"""
        results = self.model(image_path)
        boxes = results[0].boxes
        if len(boxes) > 0:
            # 按置信度排序，获取最高的一个
            sorted_indices = boxes.conf.argsort(descending=True)
            best_index = sorted_indices[0]
            best_cls = int(boxes.cls[best_index])
            best_conf = float(boxes.conf[best_index])
            class_name = self.model.names[best_cls]
            return class_name, best_conf, results[0]
        return "nan", 0.0, results[0]

    def save_annotated_image(self, result, save_path):
        """保存带标注的图片"""
        single_result = deepcopy(result)
        if len(single_result.boxes) > 0:
            single_result.boxes = single_result.boxes[0:1]  # 只保留最佳目标
        single_result.save(save_path)

# 请求处理类，管理队列和请求处理
class RequestHandler:
    def __init__(self, yolo_service):
        """初始化队列和YOLO服务"""
        self.yolo_service = yolo_service
        self.request_queue = queue.Queue()
        self.lock = threading.Lock()
        self.processing = False
        # 启动队列处理线程
        self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.worker_thread.start()

    def handle_request(self, image_file):
        """处理POST请求"""
        if self.request_queue.qsize() > 50:
            return jsonify({"class": "busy", "conf": 0.0})
        
        # 保存临时图片
        temp_path = "/tmp/temp_image.jpg"
        image_file.save(temp_path)
        
        # 将请求加入队列
        result_queue = queue.Queue()
        self.request_queue.put((temp_path, result_queue))
        
        # 等待结果
        class_name, conf = result_queue.get()
        return jsonify({"class": class_name, "conf": f"{conf:.4f}"})

    def process_queue(self):
        """处理队列中的请求"""
        while True:
            with self.lock:
                if self.processing or self.request_queue.empty():
                    continue
                self.processing = True
            
            # 从队列中取出请求
            image_path, result_queue = self.request_queue.get()
            
            # 运行推理
            class_name, conf, result = self.yolo_service.infer(image_path)
            
            # 保存带标注的图片
            date_str = datetime.now().strftime("date%Y%m%d")
            save_dir = f"/home/whs/PoseDetection/SittingWatch/logs/{date_str}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"annotated_{datetime.now().strftime('%H%M%S')}.jpg")
            self.yolo_service.save_annotated_image(result, save_path)
            
            # 返回结果
            result_queue.put((class_name, conf))
            
            # 清理临时文件
            os.remove(image_path)
            
            with self.lock:
                self.processing = False
