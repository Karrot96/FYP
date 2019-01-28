import numpy as np
from scipy import spatial
import cv2
import time

def nearestneighbours(plot,points):
    tree = spatial.cKDTree(plot)
    _,indexes1 = tree.query(points, k=[2])
    _,indexes2 = tree.query(points, k=[3])
    return indexes1, indexes2


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,60)
    shoelace = np.nonzero(edges)
    print(shoelace)
    combine = np.dstack([shoelace[0].ravel(),shoelace[1].ravel()])[0]
    indexes1, indexes2 = nearestneighbours(combine,combine)
    print(indexes1)
    print(indexes2)
    for i in range(0,len(shoelace[0])):
        cv2.line(frame, (shoelace[1][indexes1[i][0]],shoelace[0][indexes1[i][0]]), (shoelace[1][i],shoelace[0][i]),(0,255,0), 5)
        cv2.line(frame, (shoelace[1][indexes2[i][0]],shoelace[0][indexes2[i][0]]), (shoelace[1][i],shoelace[0][i]),(0,255,0), 5)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(3)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
