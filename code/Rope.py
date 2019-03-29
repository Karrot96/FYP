import cv2
import logging as log
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class Rope:
    """Rope object to store actual data nd physics model of the shoelace
    Variables:
        NO_NODES  {int} -- No. of nodes in the rope
        lace {np.array} -- location of all the nodes in 3-D space
        DISTANCE_BETWEEN_NODES {int} -- Distance between nodes in pixels
    """
    NO_NODES = 10
    # lace is shape (x,y,z)
    lace = np.zeros((NO_NODES,3))
    DISTANCE_BETWEEN_NODES = 10
    def __init__(self):
        """Rope object to store actual data nd physics model of the shoelace
        """

        log.info("Initialising rope")
        for i in range(0,len(self.lace)):
            self.lace[i] = np.array([1, 1 ,-1])
        log.debug("Finsihed rope initialisation")

    def draw_point(self, frame):
        """Draw points to signify the location of the rope nodes on the image
        
        Arguments:
            frame {np.array} -- numpy array of the image to be drawn over
        
        Returns:
            np.array -- returns the new image
        """

        for i in range(0, self.NO_NODES-1):
            cv2.circle(frame, (int(self.lace[i][0]),int(self.lace[i][1])), 2, (0,255,0), 5)
        return frame
    def draw_lace(self, frame):
        """Draw lines to signify the location of the rope nodes and links on the image
        
        Arguments:
            frame {np.array} -- numpy array of the image to be drawn over
        
        Returns:
            np.array -- returns the new image
        """
        for i in range(0, self.NO_NODES-1):
            cv2.line(frame, (int(self.lace[i][0]),int(self.lace[i][1])),(int(self.lace[i+1][0]),int(self.lace[i+1][1])), (0,255,0), 5)
        return frame