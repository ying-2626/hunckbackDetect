import os
import cv2
import numpy as np
import yaml

class DataAugmenter:
    def __init__(self, yaml_file, output_dir):
        # 加载YAML配置文件
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        self.path = config['path']
        self.train_images = os.path.join(self.path, config['train'])
        self.val_images = os.path.join(self.path, config['val'])
        self.test_images = os.path.join(self.path, config['test'])
        self.train_labels = self.train_images.replace('images', 'labels')
        self.val_labels = self.val_images.replace('images', 'labels')
        self.test_labels = self.test_images.replace('images', 'labels')
        # 设置新的保存路径为指定的输出目录
        self.new_path = output_dir
        self.new_train_images = os.path.join(self.new_path, 'train', 'images')
        self.new_train_labels = os.path.join(self.new_path, 'train', 'labels')
        self.new_val_images = os.path.join(self.new_path, 'val', 'images')
        self.new_val_labels = os.path.join(self.new_path, 'val', 'labels')
        self.new_test_images = os.path.join(self.new_path, 'test', 'images')
        self.new_test_labels = os.path.join(self.new_path, 'test', 'labels')
        
        # 定义所有的增强方法和参数
        self.augmentations = [
            {'type': 'flip', 'name': 'flip'},  # 水平翻转
            {'type': 'rotate', 'angle': 10, 'name': 'rot10'},  # 旋转10°
            {'type': 'rotate', 'angle': -10, 'name': 'rot-10'},  # 旋转-10°
            {'type': 'rotate', 'angle': 20, 'name': 'rot20'},  # 旋转20°
            {'type': 'rotate', 'angle': -20, 'name': 'rot-20'},  # 旋转-20°
            {'type': 'rotate', 'angle': 30, 'name': 'rot30'},  # 旋转30°
            {'type': 'rotate', 'angle': -30, 'name': 'rot-30'},  # 旋转-30°
            {'type': 'gaussian', 'kernel': (5,5), 'name': 'gaussian'},  # 高斯模糊
            {'type': 'brightness', 'alpha': 1, 'beta': 20, 'name': 'bright1'},  # 增加亮度档位1
            {'type': 'brightness', 'alpha': 1, 'beta': 40, 'name': 'bright2'},  # 增加亮度档位2
            {'type': 'brightness', 'alpha': 1, 'beta': -20, 'name': 'dark1'},  # 降低亮度档位1
            {'type': 'brightness', 'alpha': 1, 'beta': -40, 'name': 'dark2'},  # 降低亮度档位2
            {'type': 'contrast', 'alpha': 1.2, 'beta': 0, 'name': 'contrast1'},  # 增加对比度档位1
            {'type': 'contrast', 'alpha': 1.5, 'beta': 0, 'name': 'contrast2'},  # 增加对比度档位2
            {'type': 'contrast', 'alpha': 0.8, 'beta': 0, 'name': 'contrast_low1'},  # 降低对比度档位1
            {'type': 'contrast', 'alpha': 0.5, 'beta': 0, 'name': 'contrast_low2'},  # 降低对比度档位2
            {'type': 'sharpen', 'kernel': np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]), 'name': 'sharpen1'},  # 锐化档位1
            {'type': 'sharpen', 'kernel': np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]]), 'name': 'sharpen2'},  # 锐化档位2
        ]
    
    def augment_dataset(self):
        # 遍历train、val、test数据集
        for split in ['train', 'val', 'test']:
            images_path = getattr(self, f"{split}_images")
            labels_path = getattr(self, f"{split}_labels")
            new_images_path = getattr(self, f"new_{split}_images")
            new_labels_path = getattr(self, f"new_{split}_labels")
            # 创建新目录
            os.makedirs(new_images_path, exist_ok=True)
            os.makedirs(new_labels_path, exist_ok=True)
            for image_file in os.listdir(images_path):
                image_path = os.path.join(images_path, image_file)
                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(labels_path, label_file)
                if os.path.exists(label_path):
                    # 读取图像和标签
                    image = cv2.imread(image_path)
                    with open(label_path, 'r') as f:
                        labels = [line.strip().split() for line in f.readlines()]
                        labels = [[float(x) for x in label] for label in labels]
                    for aug in self.augmentations:
                        # 对每张图像应用增强
                        aug_image, aug_labels = self.augment_image(image, labels, aug)
                        base, ext = os.path.splitext(image_file)
                        new_image_file = f"{base}_{aug['name']}{ext}"
                        new_label_file = f"{base}_{aug['name']}.txt"
                        new_image_path = os.path.join(new_images_path, new_image_file)
                        new_label_path = os.path.join(new_labels_path, new_label_file)
                        # 保存增强后的图像和标签
                        cv2.imwrite(new_image_path, aug_image)
                        with open(new_label_path, 'w') as f:
                            for label in aug_labels:
                                f.write(' '.join([str(x) for x in label]) + '\n')
    
    def augment_image(self, image, labels, aug):
        # 根据增强类型处理图像和标签
        if aug['type'] == 'flip':
            aug_image = cv2.flip(image, 1)  # 水平翻转图像
            aug_labels = self.transform_labels_for_flip(labels)  # 调整标签
        elif aug['type'] == 'rotate':
            angle = aug['angle']
            aug_image, aug_labels = self.rotate_image_and_labels(image, labels, angle)  # 旋转图像和标签
        else:
            aug_image = self.apply_augmentation(image, aug)  # 应用其他增强方法
            aug_labels = labels  # 标签保持不变
        return aug_image, aug_labels
    
    def apply_augmentation(self, image, aug):
        # 实现非几何变换的增强方法
        if aug['type'] == 'gaussian':
            return cv2.GaussianBlur(image, aug['kernel'], 0)  # 高斯模糊
        elif aug['type'] in ['brightness', 'contrast']:
            alpha = aug['alpha']
            beta = aug['beta']
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # 亮度或对比度调整
        elif aug['type'] == 'sharpen':
            kernel = aug['kernel']
            return cv2.filter2D(image, -1, kernel)  # 锐化
    
    def transform_labels_for_flip(self, labels):
        # 水平翻转时调整标签的x坐标
        return [[cls, 1 - x_center, y_center, w, h] for cls, x_center, y_center, w, h in labels]
    
    def rotate_image_and_labels(self, image, labels, angle):
        # 旋转图像和标签
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        aug_labels = self.transform_labels_for_rotation(labels, angle, w, h)
        return aug_image, aug_labels
    
    def transform_labels_for_rotation(self, labels, angle, img_w, img_h):
        # 旋转时调整标签坐标
        new_labels = []
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cx, cy = img_w / 2, img_h / 2
        for label in labels:
            cls, x_center, y_center, w, h = label
            x = x_center * img_w
            y = y_center * img_h
            box_w = w * img_w
            box_h = h * img_h
            # 计算边界框的四个角点
            points = [
                (x - box_w/2, y - box_h/2),
                (x + box_w/2, y - box_h/2),
                (x - box_w/2, y + box_h/2),
                (x + box_w/2, y + box_h/2)
            ]
            rotated_points = []
            # 对每个角点应用旋转变换
            for px, py in points:
                px -= cx
                py -= cy
                new_px = px * cos_theta - py * sin_theta + cx
                new_py = px * sin_theta + py * cos_theta + cy
                rotated_points.append((new_px, new_py))
            # 找到旋转后的新边界框
            xs = [p[0] for p in rotated_points]
            ys = [p[1] for p in rotated_points]
            new_x_min = max(min(xs), 0)
            new_x_max = min(max(xs), img_w)
            new_y_min = max(min(ys), 0)
            new_y_max = min(max(ys), img_h)
            if new_x_max > new_x_min and new_y_max > new_y_min:
                new_w = (new_x_max - new_x_min) / img_w
                new_h = (new_y_max - new_y_min) / img_h
                new_x_center = (new_x_min + new_x_max) / 2 / img_w
                new_y_center = (new_y_min + new_y_max) / 2 / img_h
                new_labels.append([cls, new_x_center, new_y_center, new_w, new_h])
        return new_labels

# 主函数入口
if __name__ == "__main__":
    yaml_file = "/home/whs/PoseDetection/SittingWatch/buildDataset/sitting_pose/data.yaml"
    output_dir = "/home/whs/PoseDetection/SittingWatch/buildDataset/sitting_pose_augmented"
    augmenter = DataAugmenter(yaml_file, output_dir)
    augmenter.augment_dataset()
