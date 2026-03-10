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

def rotate_scale_translate(image, angle, scale, tx, ty):
    """
    同时对图像进行旋转、缩放和平移。
    :param image: 输入图像（BGR 或灰度）
    :param angle: 顺时针旋转角度（正值表示逆时针）
    :param scale: 缩放因子（>1 放大，<1 缩小）
    :param tx: 水平平移像素（正值向右，负值向左）
    :param ty: 垂直平移像素（正值向下，负值向上）
    :return: 经过仿射变换后的图像，大小不变
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # 1) 构造旋转+缩放矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 2) 在矩阵上加上平移量
    #    仿射矩阵 M 形如：
    #      [ a  b  tx0 ]
    #      [ c  d  ty0 ]
    M[0, 2] += tx
    M[1, 2] += ty
    # 3) 应用仿射变换
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed


def histogram_equalization(image):
    # 直方图均衡化（针对灰度或Y通道）
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def adaptive_histogram_equalization(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    对输入图像做 CLAHE 自适应直方图均衡化：
      - 彩色图：只增强 Y 通道
      - 灰度图：直接增强

    :param image: 输入图像（BGR 或 灰度）
    :param clipLimit: 对比度限制参数 1.5–3.0
    :param tileGridSize: 每个小瓦片大小 (8,8)
    :return: 增强后的图像
    """
    # 灰度图分支
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(image)

    # 彩色图分支：先转 YCrCb，只对 Y 通道做 CLAHE
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    ycrcb[:, :, 0] = clahe.apply(y)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def denoise(image):
    # 去噪（使用双边滤波）
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def sharpen(image):
    # 锐化（拉普拉斯算子）
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def gaussian_highpass(image, ksize=(0,0), sigma=10):
    """
    空域高斯高通滤波：原图 - 高斯低通
    :param image: 输入 BGR 或 灰度 图像
    :param ksize: 高斯核大小 (0,0) 表示由 sigma 自动计算
    :param sigma: 高斯核标准差（像素单位）
    :return: 同尺寸单通道高通响应（可能为负值），需自行归一化或加偏移
    """
    # 如果是 BGR，先转为灰度
    gray = image.copy()
    if len(image.shape)==3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) 低通
    lowpass = cv2.GaussianBlur(gray, ksize, sigmaX=sigma, sigmaY=sigma)
    # 2) 高通 = 原图 - 低通
    highpass = cv2.subtract(gray, lowpass)

    # 3) 将结果映射到 0–255
    #    有两种常见方式：绝对值或线性偏移
    hp_abs = cv2.convertScaleAbs(highpass)
    return hp_abs
