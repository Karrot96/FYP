import argparse
import logging as log
import time
import sys
import cv2
import numpy as np
import Video
np.set_printoptions(threshold=sys.maxsize)


def fromVideo(cap, out=None):
    frame = 0
    log.debug("resolution: %s", (int(cap.get(3)),int(cap.get(4))))
    if out:
        out = cv2.VideoWriter(out,cv2.VideoWriter_fourcc('D','I','V','X'), 10, (int(cap.get(3)),int(cap.get(4))))
    video = Video.Video(cap, out)
    start = time.perf_counter()
    ret = 1
    while(video.cap.isOpened and ret == 1):
        log.info("frame: %s", frame)
        ret = video.shoelaceFinding(frame)
        log.info(ret)
        if ret == 0:
            exit()
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        frame +=1
        end = time.perf_counter()
        log.info("Current Total time is %s resulting in frame rate of %s", end-start, frame/(end-start))
    cap.release()
    cv2.destroyAllWindows()

def fromCamera(cap, out=None):
    video = Video.Video(cap, out)
    frame=0
    while(True):
        video.shoelaceFinding(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame +=1
    cap.release()
    cv2.destroyAllWindows()

def videoInput():  # Used to handle argument parsing, picks the correct input source or file as well as debug mode
    parser = argparse.ArgumentParser(description='Detect and track shoelaces')
    parser.add_argument("-i","--input", action="store", help="Store input camera number if different from default")
    parser.add_argument("-v","--video", action="store", help="location of the input video file")
    parser.add_argument("-d","--debug", action="store", help="Used to enter debug mode. Using argument 'console' will output to console, any other argument will be a file path for the debug and log information")
    parser.add_argument("-l","--log", action="store_true", help="Turns on log. Debug mode wont work without this being enabled")
    parser.add_argument("-r","--record", action="store", help="Records output video to filename. Only works in log mode")
    args = parser.parse_args()
    if args.log:
        if args.debug == 'console':
            log.basicConfig(level=log.DEBUG, format='%(asctime)s, %(levelname)s , %(message)s')
            log.debug("Debug turned on. Outputting to console")
        elif args.debug:
            log.basicConfig(level=log.INFO, filename=args.debug, filemode='w', format='%(asctime)s, %(levelname)s , %(message)s')
            log.debug("Debug turned on. Outputting to {}".format(args.debug))
        else:
            log.basicConfig(level=log.INFO, format='%(asctime)s, %(levelname)s , %(message)s')
            log.info("log started")
        if args.record:
            if args.input:
                log.info("Selecting Camera %s", args.input)
                fromCamera(cv2.VideoCapture(args.input),args.record)
            elif args.video:
                log.info("Using %s as video", args.video)
                fromVideo(cv2.VideoCapture(args.video),args.record)
            else:
                log.info("Selecting Camera 0")
                fromCamera(cv2.VideoCapture(0),args.record)
    if args.input:
        log.info("Selecting Camera %s", args.input)
        fromCamera(cv2.VideoCapture(args.input))
    elif args.video:
        log.info("Using %s as video", args.video)
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
