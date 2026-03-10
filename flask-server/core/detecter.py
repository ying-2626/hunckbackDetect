import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import pose


def find_upright_rotation(img, pose_instance=None):
    """
    尝试 0/90/180/270 四个方向，选出 MediaPipe 关键点最多的方向作为正立。
    pose_instance: 可传入已初始化的 mp_pose.Pose 实例，避免重复初始化
    返回最佳角度（0/90/180/270）
    """
    best_angle = 0
    best_score = -1
    if pose_instance is None:
        mp_pose = mp.solutions.pose
        pose_instance = mp_pose.Pose(static_image_mode=True)
    for angle in [0, 90, 180, 270]:
        # 旋转图片
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        cand = cv2.warpAffine(img, M, (w, h))
        # 用 MediaPipe 检测
        results = pose_instance.process(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
        score = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def detect_person_and_fit_rect(image):
    """
    1) 用 HOG+SVM 检测整张图里最大的人体窗口
    2) 对窗口内做 Canny 边缘检测，找最大连通边缘轮廓
    3) 用 minAreaRect 拟合该轮廓，得到中心(cx,cy)、(w_box,h_box)、旋转角度 θ
    返回：cx, cy, w_box, h_box, angle_deg（以水平为 0°，逆时针为正）
    """
    # —— 1) HOG 人体检测 ——
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(image, winStride=(8,8), padding=(16,16), scale=1.05)
    if len(rects)==0:
        return None
    # 取最大面积的人体框
    x,y,w_box,h_box = max(rects, key=lambda r: r[2]*r[3])
    roi = image[y:y+h_box, x:x+w_box]

    # —— 2) Canny 边缘 & 找轮廓 ——
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    # —— 3) 最小外接矩形拟合 ——
    rect = cv2.minAreaRect(cnt)
    # rect = ((cx_rel, cy_rel), (w_rect, h_rect), angle)
    (cx_rel, cy_rel), (w_rect, h_rect), angle = rect

    # —— 转到原图坐标系 ——
    cx = x + cx_rel
    cy = y + cy_rel

    # OpenCV 的 angle 定义：
    #   如果 w_rect < h_rect，angle ∈ [-90,0) 表示 顺时针(-angle)旋转
    #   否则 angle ∈ [0,90) 表示 逆时针(90-angle)旋转
    # 我们把它统一成 “相对于水平线，逆时针正”：
    if w_rect < h_rect:
        angle_deg = -angle
    else:
        angle_deg = 90 - angle

    return cx, cy, w_box, h_box, angle_deg


def compute_preproc_params_alternative(frame, target_frac=0.4):
    """
    根据关键点算出旋转、缩放、平移参数。
    target_frac: 缩放后，骨骼宽度 / 画面宽度 的目标比例
    """
    # 用 HOG+minAreaRect 得到 cx,cy, 人体框 w_box,h_box, 以及 θ
    res = detect_person_and_fit_rect(frame)
    if res is None:
        # 检测失败就退回默认
        return 0.0, 1.0, 0.0, 0.0

    cx, cy, w_box, h_box, angle = res
    h, w = frame.shape[:2]

    # 1) 旋转：用 -angle 让人体长轴（肩胯方向）水平
    rot = -angle

    # 2) 缩放：把人体宽度 w_box 缩放到画面宽度的 target_frac
    scale = (w * target_frac) / (w_box + 1e-6)

    # 3) 平移：把 cx,cy 移到画面中心
    tx = (w / 2) - cx
    ty = (h / 2) - cy

    return rot, scale, tx, ty
