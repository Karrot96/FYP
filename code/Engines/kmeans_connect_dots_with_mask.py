import numpy as np
import cv2
import Rope
from scipy import spatial
from sklearn.cluster import KMeans
from Engines.pathfinding import MaskPath
import logging as log
import sys
np.set_printoptions(threshold=sys.maxsize)
class Engine:
    """Engine class for image processing - Kmeans version
    """

    MOVECONST = 20
    def __init__(self):
        self.rope = Rope.Rope()

    def nearestneighbours(self,plot,points,k):
        """Find the nearest point to another from a point map

        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned

        Returns:
            int -- index of the closest point within plot
        """

        tree = spatial.cKDTree(plot)
        _,indexes1 = tree.query(points, k=[k])
        return indexes1

    def adjusted_mean(self,a):
        """Checks to make sure points are close enough together to be considered parts of a string

        Arguments:
            a {np.array} -- [2 points in an array to have mean found of]

        Returns:
            [float] -- [The mean of 2 points]
        """

        if a[1]-a[0]<100:
            return np.mean(a)
        else:
            return a[0]

    def get_points(self,a):
        """Returns the points of the shoelace by looking for edges

        Arguments:
            a {np.array} -- 1-D array of all points on a given pixel line

        Returns:
            np.array -- all non-zero points on a line
        """
        b = np.nonzero(a) # Get non-zero y values constituting edges
        z = b[0]
        if len(z)>0:
            log.debug("A:%s B:%s", len(a), len(z))
            log.debug("Z:%s", z)
        out = np.pad(z,(0,len(a)-len(z)),'constant',constant_values=(0,0)) #Output array same size as input array so numpy doesnt complain
        return out # Return the midpoints of each point

    def kmeans(self, plot):
        """Make nodes through kmeans clustering
        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned

        Returns:
            int -- index of the closest point within plot
        """
        # ! kmeans++ should probably be changed to the rope.lace
        locations = KMeans(n_clusters=self.rope.NO_NODES, init='k-means++').fit(plot).cluster_centers_
        log.debug("Locations are: \n %s",locations)
        return locations



    def locateY(self, a):
        """Find the y value associated with nonzero value a

        Arguments:
            a {np.array} -- [x y] of the index of the needed y value within self.lace

        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """

        y = self.lace[a[0], a[1]]
        return (a[1],y)
    def locateX(self, a):
        """Find the x value associated with nonzero value a

        Arguments:
            a {np.array} -- [x y] of the index of the needed x value within self.lace

        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """
        x = self.lace[a[0], a[1]]
        return (x,a[0])

    def run(self, edges, mask):
        """ Used to run the processing on images

        Arguments:
            edges {np.array} -- Image in a greyscale format

        Returns:
            Rope -- full upadated rope obkect
        """

        self.lace = np.apply_along_axis(self.get_points, 0, edges)
        shoelace = np.nonzero(self.lace) # Remove padded 0's
        combinedY = np.apply_along_axis(self.locateY, 1, np.transpose(shoelace))
        self.lace = np.apply_along_axis(self.get_points, 1, edges)
        shoelace = np.nonzero(self.lace) # * Transpose lace so that non-zero acts on the right axis. Transpose shoelace so that the points are outputs in array of [y x]
        combinedX = np.apply_along_axis(self.locateX, 1, np.transpose(shoelace))
        combined = np.concatenate((combinedX,combinedY))
        clusters = self.kmeans(combined[0::20])
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = MaskPath(clusters, mask)
        x,y = path.iterate()
        log.debug("x: %s", x)
        log.debug("y: %s", y)
        for i in range(0,len(self.rope.lace)):
            self.rope.lace[i]= np.array([int(x[i]),int(y[i]),self.rope.lace[i][2]])
            # np.array([int(clusters[potential[0]][0]),int(clusters[potential[0]][1]),self.rope.lace[i][2]])
        log.debug("rope: %s",self.rope.lace)
        return self.rope