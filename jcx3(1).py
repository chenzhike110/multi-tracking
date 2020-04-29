from trackers import *
import cv2
import numpy as np
import os.path as path

cap = cv2.VideoCapture('C:\\Users\\86177\\Videos\\Captures\\1.mp4')
number = 1
while(1):
    ret, frame = cap.read()
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([78, 255, 255])

    frame = cv2.resize(frame, (1000,500), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    maskgreen = cv2.inRange(hsv, lower_green, upper_green)
    masknotgreen = cv2.bitwise_not(maskgreen)
    res = cv2.bitwise_and(frame, frame, mask=maskgreen)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(res.copy(), 10, 255, cv2.THRESH_BINARY)[1]  # 通过阈值去噪【可调整】

    image, contours, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area=[]
    for c in contours:
        area.append(cv2.contourArea(c))
    maxarea=max(area)
    index=area.index(maxarea)
    c=contours[index]
    ground=np.zeros([500,1000]) #注意长宽是反的
    #获得操场边界
    for i in range(c.shape[0]):
        ground[c[i,0,1],c[i,0,0]] = 1#注意长宽是反的
    cv2.GaussianBlur(ground, (3,3), 0)
    # 填充操场内部
    for i in range(ground.shape[0]):
        leftedge = 0
        startcolor = 0
        rightedge = 0
        for j in range(ground.shape[1]):
            if(ground[i,j]==1 and leftedge==0 ):
                leftedge=j
                rightedge= ground.shape[1]
                break
        for j in range(ground.shape[1]-1,0,-1):
            if(ground[i,j]==1 ) :
                rightedge = j
                break
        for j in range(leftedge,rightedge):
            ground[i,j]=255

    #在操场范围内定位球员
    res = cv2.bitwise_and(frame, frame, mask=masknotgreen)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(res.copy(), 10, 255, cv2.THRESH_BINARY)[1]  # 通过阈值去噪【可调整】
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)  # 对图像白色部分缩小，去噪【可调整】
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=3)  # 将白色部分膨胀恢复利于检测【可调整】
    ground=np.array(ground,np.uint8)
    res = cv2.bitwise_and(res, res, mask=ground)

    image, contours_person, hier = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for d in contours_person:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(d)
        # 计算矩形框的面积
        area_person = cv2.contourArea(d)
        if 30 < area_person < 3000:#识别区域的大小【可调整】
            detections.append([int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2),0])

    if number == 1:
        sort = tracker(detections)
        number += 1
    else:
        sort.update(detections)
    img = sort.get_information(draw=True,img=frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    cv2.imshow('result',img)

cap.release()
cv2.destroyAllWindows()




