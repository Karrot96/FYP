"""Using the mask with a Kmeans based system
"""

import sys
import logging as log
import numpy as np
import rope
from sklearn.cluster import MiniBatchKMeans
np.set_printoptions(threshold=sys.maxsize)

class Engine:
    """[summary]
    """

    def __init__(self):
        self.rope = rope.Rope()
        self.lace = None

    def kmeans(self, plot):
        """Make nodes through kmeans clustering
        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the
                                points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned

        Returns:
            int -- index of the closest point within plot
        """
        # ! kmeans++ should probably be changed to the rope.lace
        locations = KMeans(
            n_clusters=self.rope.NO_NODES,
            init='k-means++',
            batch_size=10,
            computer_labels=False
            ).fit(plot).cluster_centers_
        log.debug("Locations are: \n %s", locations)
        return locations
    
    def run(self, edges, mask):
        """[summary]
        
        Arguments:
            edges {[type]} -- [description]
            mask {[type]} -- [description]
        """
        clusters = self.kmeans(mask)
    