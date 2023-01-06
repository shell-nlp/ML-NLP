import cv2 as cv


# 检测函数
def face_detect_demo(img):
    # img = cv.resize(img, dsize=(500, 500))
    # 1 转换为灰度图
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载分类器
    face_detect = cv.CascadeClassifier(
        r"D:\install\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"
    )
    face = face_detect.detectMultiScale(gary, 1.1, 5)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow("result", img)


# 读取图片
img = cv.imread("images/2.jpg")
face_detect_demo(img)
# 等待
while True:
    if ord("q") == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
