import numpy as np
from scipy import spatial
import cv2
import time

class Rope:
    NO_NODES = 100
    lace = np.zeros((NO_NODES,3))
    DISTANCE_BETWEEN_NODES = 10
    def __init__(self):
        for i in range(0,len(self.lace)):
            self.lace[i] = np.array([i*self.DISTANCE_BETWEEN_NODES, 50 ,-1])

def nearestneighbours(plot,points):
    tree = spatial.cKDTree(plot)
    _,indexes1 = tree.query(points, k=[2])
    _,indexes2 = tree.query(points, k=[3])
    return indexes1, indexes2

rope = Rope()
cap = cv2.VideoCapture(0)
def get_mid(a):
    if np.count_nonzero(a) >0:
        return int(round(np.mean(np.nonzero(a))))
    else:
        return 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,60)
    cv2.imshow('frame',edges)
    lace = np.apply_along_axis(get_mid, 0, edges)
    shoelace = np.nonzero(lace)
    jump = int(np.count_nonzero(shoelace)/rope.NO_NODES)
    print(jump)
    for i in range(0, rope.NO_NODES-1):
        cv2.line(frame, (shoelace[0][i*jump], lace[shoelace[0][i*jump]]),(shoelace[0][i*jump+jump], lace[shoelace[0][i*jump+jump]]), (0,255,0), 5)
    cv2.imshow('edges', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
