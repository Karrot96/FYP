import numpy as np
from scipy import spatial
import cv2
import time

class Rope:
    NO_NODES = 100
    lace = np.zeros((NO_NODES,3))

def nearestneighbours(plot,points):
    tree = spatial.cKDTree(plot)
    _,indexes1 = tree.query(points, k=[2])
    _,indexes2 = tree.query(points, k=[3])
    return indexes1, indexes2

rope = Rope()
cap = cv2.VideoCapture(1)
def get_mid(a):
    # print(a.shape)
    # print(len(np.nonzero(a)))
    if np.count_nonzero(a) >0:
        # print(np.mean(np.nonzero(a)))
        return int(round(np.mean(np.nonzero(a))))
    else:
        return 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([50,50,50])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)

    # cv2.imshow('edges', frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,50,60)
    # # shoelace = np.nonzero(edges)
    # # # print(shoelace)
    # # combine = np.dstack([shoelace[0].ravel(),shoelace[1].ravel()])[0]
    # # indexes1, indexes2 = nearestneighbours(combine,combine)
    # # # print(indexes1)
    # # # print(indexes2)
    # # for i in range(0,len(shoelace[0])):
    # #     cv2.line(edges, (shoelace[1][indexes1[i][0]],shoelace[0][indexes1[i][0]]), (shoelace[1][i],shoelace[0][i]),(255,255,255), 15)
    # #     cv2.line(edges, (shoelace[1][indexes2[i][0]],shoelace[0][indexes2[i][0]]), (shoelace[1][i],shoelace[0][i]),(255,255,255), 15)
    # # Display the resulting frame
    # # edges = cv2.Canny(edges,150,200)
    # # cv2.imshow('frame',edges)
    # # print(edges.nonzero)
    lace = np.apply_along_axis(get_mid, 0, res)
    print(lace.shape)
    lace = lace[:,0]
    print(lace.shape)
    print(np.shape(lace))
    shoelace = np.nonzero(lace)
    print(np.count_nonzero(shoelace))
    jump = int(np.count_nonzero(shoelace)/rope.NO_NODES)
    print(jump)
    print(np.shape(shoelace))
    print((shoelace[0][0*jump], lace[shoelace[0][0*jump]]))
    for i in range(0, rope.NO_NODES-1):
        cv2.line(frame, (shoelace[0][i*jump], lace[shoelace[0][i*jump]]),(shoelace[0][i*jump+jump], lace[shoelace[0][i*jump+jump]]), (0,255,0), 5)
    cv2.imshow('edges', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(3)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
