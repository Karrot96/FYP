import sys
import math
import logging as log
import numpy as np
import rope
from sklearn.cluster import MiniBatchKMeans
from scipy import spatial
import joblib
from Engines.path_finding import Paths
from scipy.optimize import linear_sum_assignment
np.set_printoptions(threshold=sys.maxsize)

class Path:
    def __init__(self, points, lace):
        self.points = points
        self.rope = lace
        self.arr = [[None]*len(points)]*len(points)
    
    def generate_array(self):
        for i, j in enumerate(self.points):
            self.arr[i] = np.linalg.norm(self.points-j, axis=1)
        log.info("arr: \n %s", self.arr)
    
    def solve(self):
        self.generate_array()
        ordered = [None]*len(self.points)
        row_ind, col_ind = linear_sum_assignment(self.arr)
        for i,j in enumerate(col_ind):
            col_ind.inex(i)
        
        log.info("connections: %s", col_ind)
    

class Engine:
    """[summary]
    """

    def __init__(self):
        self.rope = rope.Rope()
        self.first = True
        self.lace = None

    def kmeans(self, plot, last):
        """Make nodes through kmeans clustering
        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the
                                points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned
        Returns:
            int -- index of the closest point within plot
        """
        locations = MiniBatchKMeans(
            n_clusters=self.rope.NO_NODES,
            init='k-means++',
            batch_size=int(self.rope.NO_NODES/3)+5,
            compute_labels=False
        ).fit(plot).cluster_centers_
        log.debug("Locations are: \n %s", locations)
        return locations

    def nearestneighbours(self, plot, points, k):
        """Find the nearest point to another from a point map

        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all
                                   the points
            points {tuple} -- (x,y) of the point to be found within the
                              plot
            k {int} -- The number of the closest point to be returned

        Returns:
            int -- index of the closest point within plot
        """

        tree = spatial.cKDTree(plot)
        _, indexes1 = tree.query(points, k=[k])
        return indexes1

    def adjusted_mean(self, arr):
        """Checks to make sure points are close enough together to be
        considered parts of a string
        Arguments:
            arr {np.array} -- [2 points in an array to have mean found of]
        Returns:
            [float] -- [The mean of 2 points]
        """
        if arr[1]-arr[0] < 100:
            return np.mean(arr)
        else:
            return arr[0]

    def get_points(self, arr):
        """Returns the points of the shoelace by looking for edges
        Arguments:
            a {np.array} -- 1-D array of all points on a given pixel line
        Returns:
            np.array -- all non-zero points on a line
        """
        non_zero = np.nonzero(arr)  # Get non-zero y values constituting edges
        first_non_zero = non_zero[0]
        if first_non_zero.any():
            log.debug("A:%s B:%s", len(arr), len(first_non_zero))
            log.debug("Z:%s", first_non_zero)
        # Output array same size as input array so numpy doesnt complain
        out = np.pad(
            first_non_zero,
            (0, len(arr)-len(first_non_zero)),
            'constant',
            constant_values=(0, 0)
            )
        return out  # Return the midpoints of each point

    def run(self, mask):
        """ Used to run the processing on images
        Arguments:
            edges {np.array} -- Image in a greyscale format
        Returns:
            Rope -- full upadated rope obkect
        """
        log.debug(self.rope.lace)
        clusters = self.kmeans(np.transpose(np.nonzero(mask)), self.rope.lace[:,:2])
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = Path(clusters, self.rope.lace[:,:2])
        log.info("rope: \n %s", self.rope.lace[:,:2])
        # if self.first:
        #     log.info("none")
        #     y_locations, x_locations = path.iterate(None)
        #     self.first = False
        # else:
        y_locations, x_locations = path.solve()
        log.debug("x: %s", x_locations)
        log.debug("y: %s", y_locations)
        for i, (j, k) in enumerate(zip(x_locations, y_locations)):
            if i > 0:
                self.rope.lace[i-1] = np.array([
                    int(j),
                    int(k),
                    self.rope.lace[i-1][2]
                    ])
        log.debug("rope: %s", self.rope.lace)
        return self.rope