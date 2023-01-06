import cv2 as cv

# 读取图片
img = cv.imread("images/1.jpg")
# 显示图片
cv.imshow("Liuyu", img)
# 等待
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
