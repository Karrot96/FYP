"""Module to Handle the Video input and frame manipulation
Used for the majority of video captures
"""
import sys
import logging as log
import numpy as np
import cv2
import models.rope
from Engines.mask_mapping import Engine
np.set_printoptions(threshold=sys.maxsize)


class Video:
    """Class handling the processing on the images

        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or
                                        camera]
            out {string} -- [Output file for the video if being outputted]
    """

    def __init__(self, cap, out):
        """Used to initialise the Video Class

        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or
                                        camera]
            out {string} -- [Output file for the video if being outputted]
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
            frame_no {int} -- The current frame number
        """
        # General error catching used to make life easier.
        # Although this is not perfect
        try:
            ret, frame = self.cap.read()
            width = frame.shape[0]
            log.debug("Frame :%s, width: %s", frame_no, width)
            log.debug(
                "Frame :%s, image resolution is: %s",
                frame_no,
                frame.shape
                )
            
            # Masking of the image
            mask_lower = np.array([220, 220, 220])
            mask_upper = np.array([255, 255, 255])
            mask = cv2.inRange(frame, mask_lower, mask_upper)
            edges = cv2.Canny(mask, 50, 60)
            log.debug("Frame :%s, Edges \n: %s", frame_no, edges)
            
            # Engines
            log.info("Starting Engine Run")
            self.rope = self.engine.run(mask)
            log.info("Engine Finished")
            frame = self.rope.draw_lace(frame)

            if self.out:
                self.out.write(frame)
                log.info("Writing Frame")
            log.debug(
                "Frame :%s, Rope positions, %s",
                frame_no,
                self.rope.lace
                )
            return ret
        except Exception as err:
            log.error("Exception occurred \n %s", err, exc_info=True)
            exit()
