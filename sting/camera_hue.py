import numpy as np
from scipy import spatial
import cv2
import time

class Rope:
    NO_NODES = 10
    lace = np.zeros((NO_NODES,3))
    DISTANCE_BETWEEN_NODES = 10
    def __init__(self):
        for i in range(0,len(self.lace)):
            self.lace[i] = np.array([i*self.DISTANCE_BETWEEN_NODES, 50 ,-1])

    def draw_lace(self, frame):
        for i in range(0, self.NO_NODES-1):
            cv2.line(frame, (int(self.lace[i][0]),int(self.lace[i][1])),(int(self.lace[i+1][0]),int(self.lace[i+1][1])), (0,255,0), 5)
        return frame

def nearestneighbours(plot,points,k):
    tree = spatial.cKDTree(plot)
    _,indexes1 = tree.query(points, k=[k])
    return indexes1

def get_mid(a):
    if np.count_nonzero(a) >0:
        return int(round(np.mean(np.nonzero(a))))
    else:
        return 0

rope = Rope()
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([200,200,200])
    upper_yellow = np.array([255,255,255])
    mask = cv2.inRange(frame, lower_yellow, upper_yellow)
    edges = cv2.Canny(mask, 50,60)
    lace = np.apply_along_axis(get_mid, 0, edges)
    shoelace = np.nonzero(lace)
    print(lace)
    jump = int(np.count_nonzero(shoelace)/rope.NO_NODES)
    laceNew = lace[lace != 0][0::jump]
    points = shoelace[0][0::jump]
    pointsCombed = np.vstack((points, laceNew)).T
    print(pointsCombed.shape)
    cv2.imshow('frame2',mask)
    map = np.zeros(rope.NO_NODES+1)
    for i in range(0, rope.NO_NODES):
        print(i)
        for k in range(1,rope.NO_NODES+1):
            potentials = nearestneighbours(pointsCombed, (rope.lace[i][0],rope.lace[i][1]), k) #Find nearest matches to the point on camera from string model
            if map[potentials[0]] == 0:
                map[potentials[0]] = 1
                rope.lace[i] = np.array([points[i],laceNew[i], rope.lace[i][2]])
                print(rope.lace[i])
                print([points[i],laceNew[i]])
                break
    print(laceNew[5]-rope.lace[5][2])
    frame = rope.draw_lace(frame)
    cv2.imshow('edges', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
