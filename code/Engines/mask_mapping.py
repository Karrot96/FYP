"""Using the mask with a Kmeans based system
"""

import sys
import logging as log
import numpy as np
import rope
from sklearn.cluster import MiniBatchKMeans
from scipy import spatial
from Engines.path_finding import Paths
np.set_printoptions(threshold=sys.maxsize)


class Path:
    """
    """

    def __init__(self, points, mask):
        """[summary]
        Arguments:
            points {[type]} -- (y,x)
            mask {[type]} -- [description]
        """
        self.MASK_WEIGHT = 5
        self.points = points
        self.mask = mask

    def flip_x(self, x1, x2, mask):
        if x1 > x2:
            return np.flip(mask, axis=1)
        return mask

    def flip_y(self, y1, y2, mask):
        if y1 > y2:
            return np.flip(mask, axis=0)
        return mask

    def mask_distance(self, p1, p2):
        """[summary]

        Arguments:
            p1 {[type]} -- [description]
            p2 {[type]} -- [description]
        """
        if p1 == p2:
            return (-255*4000)
        x_diff = abs(p1[1]-p2[1])
        y_diff = abs(p1[0]-p2[0])
        if y_diff > x_diff:
            mask = mask.T
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            x_diff = abs(p1[0]-p2[0])
            y_diff = abs(p1[1]-p2[1])
        else:
            y1 = p1[0]
            x1 = p1[1]
            y2 = p2[0]
            x2 = p2[1]
        mask = flip_x(x1, x2, flip_y(y1, y2, np.copy(self.mask)))
        x_len = len(mask[0])
        mask_flat = np.ravel(mask)
        total = 0
        for i in range(0, round(x_diff/y_diff)):
            values = mask_flat[y1*x_len+x1+i:y2*x_len+1+i:x_len+1]
            log.debug("Mask Values on iter %s: \n %s", i, values)
            total += np.sum(values)
        return total

    def algorithm(self, start, new, search_space):
        if search_space.any():
            distances = np.linalg.norm(search_space-start, axis=1)
            mask_values = np.apply_along_axis(
                self.mask_distance,
                1,
                search_space,
                start
            )
            # Mask returns 255 when detected space exsists
            # Inverting so returns 0 when line is all in mask
            # Adjusting to resonable scale
            mask_distance = abs(mask_values - 255)/255
            # Assigning a weight based on how far apart the points
            # are and the time spend outside mask
            total_weighted_distance = distances+self.MASK_WEIGHT*mask_distance
            index = np.argmin(total_weighted_distance)
            new = np.append(
                new,
                [[search_space[index], total_weighted_distance[index]]],
                axis=0
            )
            to_return = self.algorithm(
                search_space[index],
                new,
                np.delete(search_space, index, 0)
            )
            return to_return
        else:
            return (np.sum([:, 1]), new)

    def iterate(self):
        possibile_paths = [None for _ in range(len(self.points))]
        path_lengths = np.copy(possibile_paths)
        for i, point in enumerate(self.points):
            new = np.array([[self.points, 0]])
            path_lengths[i], possibile_paths[i] = self.algorithm(
                point,
                new,
                np.delete(self.points, i, 0)
            )
        index = np.argmin(path_lengths)
        x_values = []
        y_values = []
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        for best in possibile_paths[index]:
            x_values.append(best[0])
            y_values.append(best[1])
        return (x_values, y_values)
        # np.apply_along_axis(self.algorithm, 1, new, [])


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
        locations = MiniBatchKMeans(
            n_clusters=self.rope.NO_NODES,
            init='k-means++',
            batch_size=int(self.rope.NO_NODES/3),
            compute_labels=False
        ).fit(plot).cluster_centers_
        log.debug("Locations are: \n %s", locations)
        return locations

    # def run(self, edges, mask):
    #     """[summary]

    #     Arguments:
    #         edges {[type]} -- [description]
    #         mask {[type]} -- [description]
    #     """
    #     clusters = self.kmeans(mask)

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

    # def locate_y(self, arr):
    #     """Find the y value associated with nonzero value a

    #     Arguments:
    #         a {np.array} -- [x y] of the index of the needed y value within
    #                      self.lace

    #     Returns:
    #         [tuple] -- (x,y) of the location within frame of the point
    #     """

    #     y_location = self.lace[arr[0], arr[1]]
    #     return (arr[1], y_location)

    # def locate_x(self, arr):
    #     """Find the x value associated with nonzero value a

    #     Arguments:
    #         a {np.array} -- [x y] of the index of the needed x value within
    #                      self.lace

    #     Returns:
    #         [tuple] -- (x,y) of the location within frame of the point
    #     """
    #     x_location = self.lace[arr[0], arr[1]]
    #     return (x_location, arr[0])

    def run(self, mask):
        """ Used to run the processing on images

        Arguments:
            edges {np.array} -- Image in a greyscale format

        Returns:
            Rope -- full upadated rope obkect
        """
        # self.lace = np.apply_along_axis(self.get_points, 0, edges)
        # shoelace = np.nonzero(self.lace)  # Remove padded 0's
        # combined_y = np.apply_along_axis(
        #     self.locate_y,
        #     1,
        #     np.transpose(shoelace)
        #     )
        # self.lace = np.apply_along_axis(self.get_points, 1, edges)
        # # * Transpose lace so that non-zero acts on the right axis.
        # * Transpose shoelace so that the points are outputs in array of [yx]
        # shoelace = np.nonzero(self.lace)
        # combined_x = np.apply_along_axis(
        #     self.locate_x,
        #     1,
        #     np.transpose(shoelace)
        #     )
        # combined = np.concatenate((combined_x, combined_y))
        clusters = self.kmeans(np.transpose(np.nonzero(mask)))
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = Paths(clusters)
        y_locations, x_locations = path.iterate()
        log.debug("x: %s", x_locations)
        log.debug("y: %s", y_locations)
        for i, (j, k) in enumerate(zip(x_locations, y_locations)):
            self.rope.lace[i] = np.array([
                int(j),
                int(k),
                self.rope.lace[i][2]
                ])
        log.debug("rope: %s", self.rope.lace)
        return self.rope
