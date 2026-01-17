from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# 1. 加载训练好的模型
model_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\YOLOv8\runs\detect\train3\weights\best.pt"
model = YOLO(model_path)

# 2. 输入图片路径
image_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\test\test1.jpg"

# 3. 执行推理
results = model(image_path)

# 4. 获取检测结果
boxes = results[0].boxes
if len(boxes) > 0:
    # 按置信度排序，获取最高的一个
    sorted_indices = boxes.conf.argsort(descending=True)
    best_index = sorted_indices[0]
    
    # 获取最佳目标信息
    best_box = boxes.xyxy[best_index].tolist()
    best_cls = int(boxes.cls[best_index])
    best_conf = float(boxes.conf[best_index])
    class_name = model.names[best_cls]
    
    print(f"检测到的最佳目标: {class_name}, 置信度: {best_conf:.4f}, 位置: {best_box}")
    
    # 创建仅包含最佳目标的新结果
    from copy import deepcopy
    single_result = deepcopy(results[0])
    single_result.boxes = boxes[best_index:best_index+1]
    
    # 可视化和保存结果
    annotated_img = single_result.plot()
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_img[:, :, ::-1])
    plt.axis('off')
    
    save_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\test\test1_annotated.jpg"
    single_result.save(save_path)
    print(f"带标注的图片已保存至: {save_path}")
else:
    print("未检测到任何目标")



# 可以自定义推理参数
# results = model(
#     image_path,
#     conf=0.5,       # 置信度阈值，默认0.25
#     iou=0.5,        # NMS IoU阈值，默认0.7
#     max_det=100,    # 最大检测目标数，默认100
#     device=0        # 推理设备，0表示GPU，-1表示CPU
# )
