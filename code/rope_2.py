"""Module used to store the rope structure
"""

import sys
import math
import logging as log
import numpy as np
import cv2
np.set_printoptions(threshold=sys.maxsize)


class Rope:
    """Rope object to store actual data nd physics model of the shoelace
    Variables:
        NO_NODES  {int} -- No. of nodes in the rope
        lace {np.array} -- location of all the nodes in 3-D space
        DISTANCE_BETWEEN_NODES {int} -- Distance between nodes in pixels
    """
    NO_NODES = 10
    # added comment
    # lace is shape (x,y,z)
    lace = np.zeros((NO_NODES, 3))
    new = []
    DISTANCE_BETWEEN_NODES = 15

    def __init__(self):
        """Rope object to store actual data nd physics model of the shoelace
        """
        log.debug("Initialising rope")
        for i in range(0, len(self.lace)):
            self.lace[i] = np.array([1, 1, -1])
        log.debug("Finsihed rope initialisation")

    def draw_point(self, frame):
        """Draw points to signify the location of the rope nodes on the image
        Arguments:
            frame {np.array} -- numpy array of the image to be drawn over
        Returns:
            np.array -- returns the new image
        """

        for i in range(0, self.NO_NODES-1):
            cv2.circle(
                frame,
                (int(self.lace[i][0]), int(self.lace[i][1])),
                2,
                (0, 255, 0),
                5
                )
        return frame

    def draw_lace(self, frame):
        """Draw lines to signify the location of the rope nodes and links on the image

        Arguments:
            frame {np.array} -- numpy array of the image to be drawn over

        Returns:
            np.array -- returns the new image
        """
        for i in range(0, self.NO_NODES-1):
            cv2.line(
                frame,
                (int(self.lace[i][0]), int(self.lace[i][1])),
                (int(self.lace[i+1][0]), int(self.lace[i+1][1])),
                (0, 255, 0),
                5
                )
        return frame

    def find_lace(self, x_sorted, y_sorted):
        if not (len(x_sorted) == len(y_sorted)):
            raise Exception("%s: Should be an equal number of x and y values",
                            __name__
            )
        carry = 0
        for i in range(len(x_sorted)):
            # TODO: Make connections parabolics
            # TODO: Tidy up the loops          
            x_distance = x_sorted[i+1] - x_sorted[i]
            y_distance = y_sorted[i+1] - y_sorted[i]
            distance = math.hypot(x_distance, y_distance)
            x_vector = x_distance / abs(x_distance)
            # Force x vector to have abs value equal to 1
            y_vector = y_distance/abs(x_distance)
            moves = int(distance/self.DISTANCE_BETWEEN_NODES)
            for j in range(moves):
                if carry == 0:
                    # From pythagarus:
                    # a^2 + b^2 = c^2
                    # c must equal to DISTANCE_BETWEEN_NODES as that is being
                    # forced
                    # a is equal to x
                    # b is equalt to y
                    # y is equal to y_vector*x
                    # Rearranging givex x^2 = c^2/(y_vector^2+1)
                    # Forced to int as pixels cannot be partial
                    move_value = int(
                        math.sqrt(
                            self.DISTANCE_BETWEEN_NODES/(1 + y_vector**2)
                        )
                    )
                    self.new.append(
                        [
                            x_sorted[i] + move_value * x_vector * j,
                            y_sorted[i] + move_value * y_vector * j,
                            -1
                        ]
                    )
                else:
                    # From pythagarus:
                    # a^2 + b^2 = c^2
                    # c must equal to carry as that is being forced
                    # a is equal to x
                    # b is equalt to y
                    # y is equal to y_vector*x
                    # Rearranging givex x^2 = c^2/(y_vector^2+1)
                    # Forced to int as pixels cannot be partial
                    move_value = int(
                        math.sqrt(
                            carry/(1 + y_vector**2)
                        )
                    )
                    self.new.append(
                            [
                                x_sorted[i] + move_value * x_vector,
                                y_sorted[i] + move_value * y_vector,
                                -1
                            ]
                        )
                    adjusted_carry = carry
                    carry = 0
            carry = (distance - adjusted_carry) % self.DISTANCE_BETWEEN_NODES

