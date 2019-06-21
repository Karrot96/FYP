"""Using the mask with a Kmeans based system
"""

import sys
import math
import logging as log
import numpy as np
import rope
from sklearn.cluster import MiniBatchKMeans
from scipy import spatial
import joblib
np.set_printoptions(threshold=sys.maxsize)


class Path:
    """Used to connect points in a path
    Arguments:
        points {np.array} -- (y,x)
        mask {np.array} -- mask image
    """
    def __init__(self, points, mask):
        """Used to connect points in a path
        Arguments:
            points {np.array} -- (y,x)
            mask {np.array} -- mask image
        """
        self.MASK_WEIGHT = 40
        self.points = points
        self.mask = mask

    def flip_x(self, x1, x2, mask):
        """Swap two points in x direction
        
        Arguments:
            x1 {int} -- x1 value
            x2 {int} -- x2 value
            mask {np.array} -- mask image
        
        Returns:
            np.array -- mask image
        """
        if x1 > x2:
            return np.flip(mask, axis=1)
        return mask

    def flip_y(self, y1, y2, mask):
        """Swap two points in y direction
        
        Arguments:
            y1 {int} -- y1 value
            y2 {int} -- y2 value
            mask {np.array} -- mask image
        
        Returns:
            np.array -- mask image
        """
        if y1 > y2:
            return np.flip(mask, axis=0)
        return mask

    def mask_distance(self, p1, p2, mask):
        """get adjusted distance from mask  

        Arguments:
            p1 {np.array} -- points 1 in (y,x)
            p2 {np.array} -- points 2 in (y,x)
        """
        x_diff = abs(p1[1]-p2[1])
        y_diff = abs(p1[0]-p2[0])
        if y_diff > x_diff:
            mask = mask.T
            x1 = int(p1[0])
            y1 = int(p1[1])
            x2 = int(p2[0])
            y2 = int(p2[1])
            x_diff = abs(p1[0]-p2[0])
            y_diff = abs(p1[1]-p2[1])
        else:
            y1 = int(p1[0])
            x1 = int(p1[1])
            y2 = int(p2[0])
            x2 = int(p2[1])
        mask = self.flip_x(x1, x2, self.flip_y(y1, y2, np.copy(self.mask)))
        x_len = len(mask[0])
        mask_flat = np.ravel(mask)
        total = 0
        if y_diff == 0:
            values = mask_flat[y1*x_len+x1:y2*x_len+1:1]
            total += np.sum(values)
        elif x_diff == 0:
            values = mask_flat[y1*x_len+x1:y2*x_len+1:x_len]
            total += np.sum(values)
        else:
            for i in range(0, int(x_diff/y_diff)):
                values = mask_flat[y1*x_len+x1+i:y2*x_len+1+i:x_len+1]
                log.debug("Mask Values on iter %s: \n %s", i, values)
                total += np.sum(values)
        return total
    
    def algorithm_loop(self, start, new, search_space):
        """Loops over the distance algorithm
        
        Arguments:
            start {np.array} -- start node location
            new {np.array} -- output path
            search_space {np.array} -- points left to search
        
        Returns:
            np.array -- shortest path
        """
        log.debug("new: \n %s", new)
        if search_space.any():
            log.debug("Start: %s", start)
            distances = np.linalg.norm(search_space-start, axis=1)
            log.debug("distances: \n %s", distances)
            mask_values = np.apply_along_axis(
                self.mask_distance,
                1,
                search_space,
                start,
                self.mask
            )
            # Mask returns 255 when detected space exsists
            # Inverting so returns 0 when line is all in mask
            # Adjusting to resonable scale
            mask_distance = abs(mask_values - 255)/255
            log.debug("masks:\n %s", mask_distance)
            # Assigning a weight based on how far apart the points
            # are and the time spend outside mask
            total_weighted_distance = distances+np.exp2(mask_distance*(1/self.MASK_WEIGHT))
            log.debug("total: \n %s", total_weighted_distance)
            index = np.argmin(total_weighted_distance)
            log.debug("index: %s", index)
            new = np.append(
                new,
                [[search_space[index], total_weighted_distance[index]]],
                axis=0
            )
            to_return = self.algorithm_loop(
                search_space[index],
                new,
                np.delete(search_space, index, 0)
            )
            return to_return
        else:
            return [np.sum(new[:, 1]), new]

    def algorithm(self, start):
        """legacy connection
        
        Arguments:
            start {np.array} -- start node location
        
        Returns:
            np.array -- the output path
        """
        return self.algorithm_loop(start, np.array([[start, 0]]), np.copy(self.points))

    def get_value(self, arr):
        """get first value of array
        
        Arguments:
            arr {np.array} -- an array
        
        Returns:
            np.array -- first element of array
        """
        return arr[0]

    def iterate(self, rope_loc):
        """iterate through all starting nodes
        
        Arguments:
            rope_loc {np.array} -- locations of the points on the shoelace
        
        Returns:
            np.array -- shortest rope path
        """
        if rope_loc is not None:
            return self.iterate_new(rope_loc)
        # possibile_paths = [None for _ in range(len(self.points))]
        # path_lengths = np.copy(possibile_paths)
        log.info("starting for loop")
        possibile_paths = joblib.Parallel(
            n_jobs=8
        )(map(
            joblib.delayed(self.algorithm),
            np.copy(self.points)
            )
        )
        # for i, point in enumerate(self.points):
        #     log.info(i)
        #     new = np.array([[point, 0]])
        #     path_lengths[i], possibile_paths[i] = self.algorithm_loop(
        #         point,
        #         new,
        #         np.delete(self.points, i, 0)
        #     )
        log.info("loop ended")
        x = np.apply_along_axis(self.get_value, 1, possibile_paths)
        index = np.argmin(x)
        x_values = []
        y_values = []
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        log.debug("paths : \n %s", possibile_paths[index][1])
        for best in possibile_paths[index][1]:
            log.debug("best : \n %s", best)
            x_values.append(best[0][0])
            y_values.append(best[0][1])
        return (x_values, y_values)

    def iterate_new(self, rope_loc):
        """iterate through all starting nodes
        
        Arguments:
            rope_loc {np.array} -- locations of the points on the shoelace
        
        Returns:
            np.array -- shortest rope path
        """
        log.info("Getting previous starting value")
        rope_xy = [rope_loc[0][0], rope_loc[0][1]]
        log.debug("xy: %s", rope_xy)
        closest_point1 = np.linalg.norm(self.points-rope_xy, axis=1)
        index = np.argmin(closest_point1)
        possibile_path1 = self.algorithm(self.points[index])
        log.info("One run")
        x_only = rope_loc[:,0]
        index = np.argmax(abs(x_only-900))
        log.info(x_only)
        rope_xy = [rope_loc[index][0], rope_loc[index][1]]
        closest_point2 = np.linalg.norm(self.points-rope_xy, axis=1)
        index = np.argmin(closest_point2)
        possibile_path2 = self.algorithm(self.points[index])
        log.info("Run 2 done")
        log.info("d1: %s, d2: %s", possibile_path2[0], possibile_path1[0])
        if possibile_path1[0]< possibile_path2[0]:
            possibile_path = possibile_path1
        else:
            possibile_path = possibile_path2
        x_values = []
        y_values = []
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        log.debug("paths : \n %s", possibile_path[1])
        for best in possibile_path[1]:
            log.debug("best : \n %s", best)
            x_values.append(best[0][0])
            y_values.append(best[0][1])
        return (x_values, y_values)
        # np.apply_along_axis(self.algorithm, 1, new, [])


class Engine:
    """Engine for mask_mapping
    """

    def __init__(self):
        """Engine for mask_mapping
        """
        self.rope = rope.Rope()
        self.first = True
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
        clusters = self.kmeans(np.transpose(np.nonzero(mask)))
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = Path(clusters, mask)
        if self.first:
            y_locations, x_locations = path.iterate(None)
            self.first = False
        else:
            y_locations, x_locations = path.iterate(self.rope.lace)
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
