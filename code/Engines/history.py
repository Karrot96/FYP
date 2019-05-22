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
        self.MASK_WEIGHT = 1
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

    def mask_distance(self, p1, p2, mask):
        """[summary]
        Arguments:
            p1 {[type]} -- [description]
            p2 {[type]} -- [description]
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
            total_weighted_distance = distances
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
        return self.algorithm_loop(start, np.array([[start, 0]]), np.copy(self.points))

    def get_value(self, arr):
        return arr[0]

    def loop(self, data, mean, arr):
        for i in range(len(arr)):
            # print("i: ", i)
            total = 0
            locs = []
            # print(data[0])
            for k, j in enumerate(data[0]):
                # print("j: ", j)
                if j == i:
                    total += 1
                    locs.append(k)
            if total > 1:
                best_val = -1
                # print(locs)
                for l in locs:
                    val = abs(mean[l]-arr[l][i])
                    # print(val)
                    if val > best_val:
                        best_val = abs(mean[l]-arr[l][i])
                        best = l
                    # print("best: ", best)
                for n, m in enumerate(data[0]):
                    if n in locs and not (n == best):
                        data[0][n] = i+1
        # print("data[0]: ", data[0])
        return data

    def new_sort(self, rope_loc):
        new = self.points
        log.info("points: \n %s", self.points)
        org = np.flip(rope_loc[:,:2], axis=1)
        log.info("org: \n %s", org)
        arr = []
        for i, j in enumerate(org):
            values = np.linalg.norm(new-j, axis=1)
            arr.append(values)
        arr = np.array(arr)
        log.info(arr)
        sorted_arg_mins = []
        for i in arr:
            sorted_arg_mins.append(np.argsort(i, kind='mergesort'))
        log.info("Sorted args :\n %s", sorted_arg_mins)
        data = np.transpose(sorted_arg_mins)
        # print(data)
        mean = np.mean(arr, axis=1)
        while (not (len(np.unique(data[0])) == len(data[0])) or np.max(data[0]) >= len(data[0])):
            data = self.loop(data, mean, arr)
            data = np.where(data >= len(data[0]), 0, data)
        log.info("data: %s", data[0])
        rope_order = [None] * len(data)
        for i,j in enumerate(data[0]):
            rope_order[i]=(self.points[j])
        log.info("rope_order: \n %s", rope_order)
        return rope_order

    def iterate(self, rope_loc):
        if rope_loc is None:
            return self.iterate_new(rope_loc)
        # possibile_paths = [None for _ in range(len(self.points))]
        # path_lengths = np.copy(possibile_paths)
        log.info("starting for loop")
        path_ordered = self.new_sort(rope_loc)
        log.info("loop ended")
        x_values = []
        y_values = []
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        log.debug("paths : \n %s", path_ordered)
        for best in path_ordered:
            log.debug("best : \n %s", best)
            x_values.append(best[0])
            y_values.append(best[1])
        return (x_values, y_values)

    def iterate_new(self, rope_loc):
        log.info("starting for loop")
        log.info("points: \n %s", self.points)
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
        # np.apply_along_axis(self.algorithm, 1, new, [])


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
        if self.first:
            locations = MiniBatchKMeans(
                n_clusters=self.rope.NO_NODES,
                init='k-means++',
                batch_size=int(self.rope.NO_NODES/3)+5,
                compute_labels=False
            ).fit(plot).cluster_centers_
            log.debug("Locations are: \n %s", locations)
            return locations
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
        clusters = self.kmeans(np.transpose(np.nonzero(mask)), self.rope.lace[:,:2])
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = Path(clusters, mask)
        log.info("rope: \n %s", self.rope.lace[:,:2])
        if self.first:
            log.info("none")
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
