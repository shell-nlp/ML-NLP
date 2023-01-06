import cv2 as cv

# 读取图片
img = cv.imread("images/1.jpg")
# TODO 修改尺寸
resize_img = cv.resize(img, dsize=(200, 200))

# 显示原图
cv.imshow("liuyu", img)
# 显示修改后图片
cv.imshow("liuyu_200_200", resize_img)
# 打印原图尺寸大小
print("未修改", img.shape)
# 打印修改后尺寸大小
print("修改后", resize_img.shape)

# 保存修改后图片
cv.imwrite("images/resize_liuyu.jpg", resize_img)

# 等待
while True:
    if ord("q") == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
