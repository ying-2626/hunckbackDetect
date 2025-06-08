import cv2
import numpy as np

def correct_perspective(image, src_points, dst_points):
    # 透视变换校正
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

def rotate_and_scale(image, angle, scale):
    # 旋转并缩放
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))

def histogram_equalization(image):
    # 直方图均衡化（针对灰度或Y通道）
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)

def denoise(image):
    # 去噪（使用双边滤波）
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def sharpen(image):
    # 锐化（拉普拉斯算子）
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

