# TODO: Fill docstring
"""[summary]

Returns:
    [type] -- [description]
"""
import sys
import logging as log
import numpy as np
import cv2
import rope
from Engines.mask_mapping import Engine
np.set_printoptions(threshold=sys.maxsize)


class Video:
    """Class handling the processing on the images

        Arguments:
            cap {cv2.VideoCapture} -- The video stream from the file or camera
            out {string} -- Output file for the video if being outputted
    """

    def __init__(self, cap, out):
        """Used to initialise the Video Class

        Arguments:
            cap {cv2.VideoCapture} -- The video stream from the file or camera
            out {string} -- Output file for the video if being outputted
        """

        log.info("Initialising Video Input")
        self.engine = Engine()
        self.cap = cap
        self.out = out
        self.rope = None
        log.debug("Finished Video Initialisation")

    def shoelace_finding(self, frame_no):
        """Find the location of the shoelace

        Arguments:
            None
        """
        try:
            ret, frame = self.cap.read()
            if not ret:
                return 0
            log.debug("Frame :%s, width: %s", frame_no, frame.shape[0])
            log.debug("Frame :%s, image resolution is: %s",
                      frame_no,
                      frame.shape
                      )
            lower_yellow = np.array([220, 220, 220])
            upper_yellow = np.array([255, 255, 255])
            mask = cv2.inRange(frame, lower_yellow, upper_yellow)
            edges = cv2.Canny(mask, 50, 60)
            log.debug("Frame :%s, Edges \n: %s", frame_no, edges)
            # Engines
            log.info("Starting Engine Run")
            self.rope = self.engine.run(edges)
            log.info("Engine Finished")
            frame = self.rope.draw_lace(frame)
            # cv2.imshow('edges', mask)
            # cv2.imshow('frame', frame)
            if self.out:
                self.out.write(frame)
                log.info("Writing Frame")
            log.debug("Frame :%s, Rope positions, %s",
                      frame_no,
                      self.rope.lace
                      )
            return 1
        except Exception as e:
            log.error("Exception occurred: %s", e, exc_info=True)
            exit()
