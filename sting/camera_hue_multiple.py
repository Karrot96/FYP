import argparse
import logging
import time

import cv2
import numpy as np
from scipy import spatial


log = logging.getLogger(__name__)

### Get the points on a give column of data
def get_points(a):
    b = np.nonzero(a) # Get non-zero y values constituting edges
    z = b[0]
    if len(z)>1: # Make sure there are edges
        z =z[0:-1] if len(z)%2!=0 else z #Adjust if odd number found
        z=z.reshape(int(len(z)/2),2) #Reshape into points
        if len(z)>1: ##incase only one edge exsists
            out=z.mean(axis=1)
            out=np.apply_along_axis(np.round,0,out) #convert from flaot 
        else:
            out=[round(z.mean())]
        out = np.pad(out,(0,len(a)-len(out)),'constant',constant_values=(0,0)) #Output array same size as input array so numpy doesnt complain
        return out # Return the midpoints of each point
    else:
        return np.pad([0],(0,len(a)-len([0])),'constant',constant_values=(0,0)) #Zero Output

def nearestneighbours(plot,points,k):
    tree = spatial.cKDTree(plot)
    _,indexes1 = tree.query(points, k=[k])
    return indexes1


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


class Video:

    def __init__(self, cap):
        log.info("initialising video")
        self.rope = Rope()
        self.cap = cap

    def shoelaceFinding(self):
        try:
            ret, frame = self.cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([150,150,150])
            upper_yellow = np.array([255,255,255])
            mask = cv2.inRange(frame, lower_yellow, upper_yellow)
            edges = cv2.Canny(mask, 50,60)
            lace = np.apply_along_axis(get_points, 0, edges)
            shoelace = (np.transpose(np.nonzero(lace.T))) #Transpose lace so that non-zero acts on the right axis. Transpose shoelace so that the points are outputs in array of [x y]
            frame = self.rope.draw_lace(frame)
            cv2.imshow('edges', edges)
            log.debug("Rope positions, %s", self.rope)
        except Exception as e:
            log.error("Exception occurred", exc_info=True)


def fromVideo(cap):
    video = Video(cap)
    while(video.cap.isOpened):
        video.shoelaceFinding()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def fromCamera(cap):
    video = Video(cap)
    while(True):
        video.shoelaceFinding()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def videoInput():  # Used to handle argument parsing, picks the correct input source or file as well as debug mode
    parser = argparse.ArgumentParser(description='Detect and track shoelaces')
    parser.add_argument("-i","--input", action="store", help="Store input camera number if different from default")
    parser.add_argument("-v","--video", action="store", help="location of the input video file")
    parser.add_argument("-d","--debug", action="store", help="Used to enter debug mode. Using argument 'console' will output to console, any other argument will be a file path for the debug and log information")
    parser.add_argument("-l","--log", action="store_true", help="Turns on log. Debug mode wont work without this being enabled")
    args = parser.parse_args()
    if args.log:
        if args.debug == 'console':
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.DEBUG)
            log.addHandler(c_handler)
            log.debug("Debug turned on. Outputting to console")
        elif args.debug:
            f_handler = logging.FileHandler(args.debug)
            f_handler.setLevel(logging.DEBUG)
            f_format = logging.Formatter("%(asctime)s, %(name)s , %(levelname)s , %(message)s")
            f_handler.setFormatter(f_format)
            log.addHandler(f_handler)
            log.debug("Debug turned on. Outputting to {}".format(args.debug))
        else:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            log.addHandler(c_handler)
            log.info("log started")
    if args.input:
        log.info("Selecting Camera %s", args.input)
        fromCamera(cv2.VideoCapture(args.input))
    elif args.video:
        log.info("Using %s as camera input", args.input)
        fromVideo(cv2.VideoCapture(args.video))
    else:
        log.info("Selecting Camera 0")
        fromCamera(cv2.VideoCapture(0))


def main():
    videoInput()

    # while(True):
    #     ret, frame = cap.read()
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     lower_yellow = np.array([150,150,150])
    #     upper_yellow = np.array([255,255,255])
    #     mask = cv2.inRange(frame, lower_yellow, upper_yellow)
    #     edges = cv2.Canny(mask, 50,60)
    #     lace = np.apply_along_axis(get_points, 0, edges)
    #     shoelace = (np.transpose(np.nonzero(lace.T))) #Transpose lace so that non-zero acts on the right axis. Transpose shoelace so that the points are outputs in array of [x y]
    #     #jump = int(np.count_nonzero(shoelace)/rope.NO_NODES)
    #     laceNew = lace[np.nonzero(lace)]
    #     print(laceNew)
    #     # print(lace)
    #     print(laceNew[30])
    #     print(lace[shoelace[30][0]][shoelace[30][1]])
    #     # pointsCombed = np.vstack((points, laceNew)).T
    #     # print(pointsCombed.shape)
    #     # cv2.imshow('frame2',mask)
    #     # map = np.zeros(rope.NO_NODES+1)
    #     # for i in range(0, rope.NO_NODES):
    #     #     print(i)
    #     #     for k in range(1,rope.NO_NODES+1):
    #     #         potentials = nearestneighbours(pointsCombed, (rope.lace[i][0],rope.lace[i][1]), k) #Find nearest matches to the point on camera from string model
    #     #         if map[potentials[0]] == 0:
    #     #             map[potentials[0]] = 1
    #     #             rope.lace[i] = np.array([points[i],laceNew[i], rope.lace[i][2]])
    #     #             print(rope.lace[i])
    #     #             print([points[i],laceNew[i]])
    #     #             break
    #     # print(laceNew[5]-rope.lace[5][2])
    #     # frame = rope.draw_lace(frame)
    #     # cv2.imshow('edges', edges)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()