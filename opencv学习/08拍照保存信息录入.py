import cv2 as cv

cap = cv.VideoCapture(0)

flag = 1
num = 1
while (cap.isOpened()):  # 检测是否在开启状态
    ret_flag, Vshow = cap.read()
    cv.imshow("Capture", Vshow)
    k = cv.waitKey(1) & 0xFF  # 按键判断
    if k == ord("s"):
        cv.imwrite(f"save/{num}.jpg", Vshow)
        print("成功 保存 图片", num)
        num = num + 1
    elif k == ord(" "):
        break
cap.release()
cv.destroyAllWindows()