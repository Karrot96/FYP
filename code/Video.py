import numpy as np
import cv2
import Rope
from scipy import spatial
import logging as log
import sys
from Engines.KmeansConnectDots import Engine
np.set_printoptions(threshold=sys.maxsize)

class Video:
    """Class handling the processing on the images

        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or camera]
            out {string} -- [Output file for the video if being outputted]
    """

    def __init__(self, cap, out):
        """Used to initialise the Video Class

        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or camera]
            out {string} -- [Output file for the video if being outputted]
        """

        log.info("Initialising Video Input")
        self.engine = Engine()
        self.cap = cap
        self.out = out
        log.debug("Finished Video Initialisation")


    def shoelaceFinding(self,frameNo):
        """Find the location of the shoelace

        Arguments:
            None
        """
        try:
            ret, frame = self.cap.read()
            if not ret:
                return 0
            log.debug("Frame :%s, width: %s",frameNo, frame.shape[0])
            log.debug("Frame :%s, image resolution is: %s",frameNo, frame.shape)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([220,220,220])
            upper_yellow = np.array([255,255,255])
            mask = cv2.inRange(frame, lower_yellow, upper_yellow)
            edges = cv2.Canny(mask, 50,60)
            log.debug("Frame :%s, Edges \n: %s", frameNo, edges)
            ##Engines
            log.info("Starting Engine Run")
            self.rope = self.engine.run(edges)
            log.info("Engine Finished")


            frame = self.rope.draw_lace(frame)
            # cv2.imshow('edges', edges)
            # cv2.imshow('frame', frame)
            if self.out:
                self.out.write(frame)
                log.info("Writing Frame")
            log.debug("Frame :%s, Rope positions, %s",frameNo, self.rope.lace)
            return 1
        except Exception as e:
            log.error("Exception occurred", exc_info=True)
            exit()
