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
    DISTANCE_BETWEEN_NODES = 5

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
        for i in range(len(self.lace)-1):
            cv2.line(
                frame,
                (int(self.lace[i][0]), int(self.lace[i][1])),
                (int(self.lace[i+1][0]), int(self.lace[i+1][1])),
                (0, 255, 0),
                5
                )
        return frame

    def find_lace(self, x_sorted, y_sorted):
        """ Create a lace along a given path
        
        Arguments:
            x_sorted {np.array} -- locations of the x values
            y_sorted {np.array} -- locations of the y values
        
        Raises:
            Exception: If input arrays are not of same length
        """
        if not len(x_sorted) == len(y_sorted):
            raise Exception("%s: Should be an equal number of x and y values",
                            __name__
                            )
        carry = 0
        for i in range(len(x_sorted)-1):
            # TODO: Make connections parabolics
            # TODO: Tidy up the loops
            x_distance = x_sorted[i+1] - x_sorted[i]
            y_distance = y_sorted[i+1] - y_sorted[i]
            distance = math.hypot(x_distance, y_distance)
            log.debug("distance: %s", distance)
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
                    adjusted_carry = 0
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
        # log.debug(self.new)
        self.lace = np.array(self.new)
        # log.debug(self.lace)

    def follow_the_leader(self, move, movement_vector, node):
        """Implements the follow the leader loop
        
        Arguments:
            move {np.array} -- Location of the node to move
            movement_vector {np.array} -- Vector along which to move node
            node {np.array} -- location of node movement is towards
        
        Returns:
            np.array -- new location of the moved node
        """
        x = math.sqrt(
            (move[0] - node[0])**2
            + (move[1] - node[1])**2
            + (move[2] - node[2])**2
        )
        while x > self.DISTANCE_BETWEEN_NODES:
            move = move+movement_vector
            x = math.sqrt(
                (move[0] - node[0])**2
                + (move[1] - node[1])**2
                + (move[2] - node[2])**2
            )
        return move

    def follow_the_leader_simple(self, move, node):
        """Handles the set up of the follow the leader
        
        Arguments:
            move {np.array} -- location of node to move
            node {np.array} -- location of previous node that was moved
        
        Returns:
            {np.array} -- new location of the move node
        """
        log.debug(move)
        log.debug(node)
        new_vector = node - move
        log.debug("new_vector:%s", new_vector)
        divisor = np.absolute(new_vector[np.argmax(np.absolute(new_vector))])
        movement_vector = new_vector / divisor
        move = move + movement_vector
        return self.follow_the_leader(move, movement_vector, node)


    # This should be done recursively would be more efficient
    def implement_follow_the_leader(self,
                                    node,
                                    position,
                                    begining,
                                    end,
                                    second=None,
                                    position_two=None
                                    ):
        """Used to handle the FTL method
        
        Arguments:
            node {int} -- current node number  
            position {np.array} -- location to move node to
            begining {bool} -- first node being moved
            end {bool} -- last node being moved
        
        Keyword Arguments:
            second {int} -- number of the second node (default: {None})
            position_two {np.array} -- location of the second node (default: {None})
        """
        log.debug("node1: %s, Node2: %s", node, second)
        # Used to handle the different states of the FTL alogirthm
        if begining:
            log.debug("Node : %s, position: %s", node, position)
            originalNode = node
            second_orig = second
            self.lace[node] = position
            self.lace[second] = position_two
            while node > 0:
                currentNode = node - 1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[node]
                )
                node = currentNode
            node = originalNode
            tmp_original = self.lace[originalNode:second_orig]
            while second > node:
                currentNode = second - 1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[second]
                )
                second = currentNode
            tmp_after_1 = self.lace[originalNode:second_orig]
            self.lace[originalNode:second_orig] = tmp_original
            node = originalNode
            while node < second_orig:
                currentNode = node+1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[node]
                )
                node = currentNode
            tmp_after_2 = self.lace[originalNode:second_orig]
            averaged = np.average(list(zip(tmp_after_1, tmp_after_2)), axis=1)
            self.lace[originalNode:second_orig] = averaged
        elif end:
            log.debug("Node : %s, position: %s", node, position)
            originalNode = node
            second_orig = second
            self.lace[node] = position
            self.lace[second] = position_two
            while second < len(self.lace)-1:
                currentNode = second + 1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[second]
                )
                second = currentNode
            second = second_orig
            tmp_original = self.lace[originalNode:second_orig]
            while second > node:
                currentNode = second - 1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[second]
                )
                second = currentNode
            tmp_after_1 = self.lace[originalNode:second_orig]
            self.lace[originalNode:second_orig] = tmp_original
            node = originalNode
            while node < second_orig:
                currentNode = node+1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[node]
                )
                node = currentNode
            tmp_after_2 = self.lace[originalNode:second_orig]
            list_vals = list(zip(tmp_after_1, tmp_after_2))
            log.debug("list: \n %s", list_vals)
            averaged = np.average(list_vals, axis=1)
            self.lace[originalNode:second_orig] = averaged
        else:
            log.debug("Node : %s, position: %s", node, position)
            originalNode = node
            second_orig = second
            self.lace[node] = position
            self.lace[second] = position_two
            tmp_original = self.lace[originalNode:second_orig]
            while second > node:
                currentNode = second - 1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[second]
                )
                second = currentNode
            tmp_after_1 = self.lace[originalNode:second_orig]
            self.lace[originalNode:second_orig] = tmp_original
            node = originalNode
            while node < second_orig:
                currentNode = node+1
                self.lace[currentNode] = self.follow_the_leader_simple(
                    self.lace[currentNode],
                    self.lace[node]
                )
                node = currentNode
            tmp_after_2 = self.lace[originalNode:second_orig]
            averaged = np.average(list(zip(tmp_after_1, tmp_after_2)), axis=1)
            self.lace[originalNode:second_orig] = averaged
