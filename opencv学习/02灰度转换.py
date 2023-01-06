import cv2 as cv

# 读取图片
img = cv.imread("images/1.jpg")
# TODO 灰度转换
gray_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

# 显示图片
cv.imshow("gray_Liuyu", gray_img)

# 保存灰度图片
cv.imwrite("images/gray_1.jpg", gray_img)

# 显示图片
cv.imshow("Liuyu", img)
# 等待
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
