import numpy as np
import cv2

cap = cv2.VideoCapture('videos/t1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    lower_white = np.array([230, 230, 230])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)
    newimg = cv2.bitwise_or(frame, frame, mask=mask)
    gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    new = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('frame', new)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()