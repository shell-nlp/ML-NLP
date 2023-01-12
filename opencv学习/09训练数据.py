import cv2 as cv
import os
from PIL import Image
import numpy as np


def getImageAndLables(path):
    # 储存人脸数据
    facesSamples = []
    # 储存姓名数据
    ids = []
    # 储存图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv.CascadeClassifier(
        r"D:\install\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"
    )
    # 遍历列表中图片
    for imagePath in imagePaths:
        # 打开图片，灰度化 PIL有九种不同模式  1 L  P  RGB RGBA  CMYK YCbCr  I F
        PIL_img = Image.open(imagePath).convert("L")

        # ----------------
        # ----------------
        # 将 图像转化为数组  以黑白深浅
        img_numpy = np.array(PIL_img, "uint8")
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split(".")[0])
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y: y + h, x:x + w])
    # 打印脸部特征和id
    print("id", id)
    print("fs", facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    path = "save/"
    # 获取图像数组和id标签数值和姓名
    faces, ids = getImageAndLables(path)
    # 加载识别器

    recongnizer = cv.face.LBPHFaceRecognizer_create()
    # 训练
    recongnizer.train(faces, np.array(ids))
    # 保存
    recongnizer.write("trainer/trainer.yml")
