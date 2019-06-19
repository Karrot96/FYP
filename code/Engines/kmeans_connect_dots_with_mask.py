"""[summary]

Returns:
    [type] -- [description]
"""
import sys
import logging as log
import numpy as np
import rope
from scipy import spatial
from sklearn.cluster import KMeans
np.set_printoptions(threshold=sys.maxsize)

class MaskPath:
    """[summary]

    Returns:
        [type] -- [description]
    """

    total_distance = -1
    best = None
    total_outside = 0
    total_checked = 0

    def __init__(self, points, mask):
        """[summary]

        Arguments:
            points {[type]} -- [description]
            mask {[type]} -- [description]
        """
        self.points = points
        self.mask = mask

    def iterate_mask(
            self,
            curr_x,
            curr_y,
            gradient,
            major_dir_x,
            dir_x,
            dir_y
    ):
        """[summary]

        Arguments:
            curr_x {[type]} -- [description]
            curr_y {[type]} -- [description]
            gradient {[type]} -- [description]
            major_dir_x {[type]} -- [description]
            dir_x {[type]} -- [description]
            dir_y {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        if major_dir_x:
            next_y = curr_y
            for i in range(gradient):
                next_x = curr_x+dir_x
                if i == gradient-1:
                    next_y = curr_y+dir_y
                if self.mask[next_y, next_x] == 0:
                    self.total_outside += 1
                self.total_checked += 1
        else:
            next_x = curr_x
            for i in range(gradient):
                next_y = curr_y+dir_y
                if i == gradient-1:
                    next_x = curr_x+dir_x
                if self.mask[next_y, next_x] == 0:
                    self.total_outside += 1
                self.total_checked += 1
        return next_x, next_y

    def check_mask(self, point1, point2):
        """[summary]

        Arguments:
            point1 {[type]} -- [description]
            point2 {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        x_diff = point1[1]-point2[1]
        y_diff = point1[0]-point2[0]
        x_diff_abs = abs(x_diff)
        y_diff_abs = abs(y_diff)
        gradient = y_diff_abs/x_diff_abs
        if gradient < 1:
            major_dir_x = False
            gradient = round(1/gradient)
        else:
            major_dir_x = True
            gradient = round(gradient)
        dir_x = int(x_diff/x_diff_abs)
        dir_y = int(y_diff/y_diff_abs)
        curr_x = point1[1]
        curr_y = point1[0]
        end_x = point2[1]
        end_y = point2[0]
        if dir_x == -1:
            if dir_y == -1:
                while (curr_x <= end_x and curr_y <= end_y):
                    curr_x, curr_y = self.iterate_mask(
                        curr_x,
                        curr_y,
                        gradient,
                        major_dir_x,
                        dir_x,
                        dir_y
                    )
            else:
                while (curr_x <= end_x and curr_y >= end_y):
                    curr_x, curr_y = self.iterate_mask(
                        curr_x,
                        curr_y,
                        gradient,
                        major_dir_x,
                        dir_x,
                        dir_y
                    )
        else:
            if dir_y == -1:
                while (curr_x >= end_x and curr_y <= end_y):
                    curr_x, curr_y = self.iterate_mask(
                        curr_x,
                        curr_y,
                        gradient,
                        major_dir_x,
                        dir_x,
                        dir_y
                    )
            else:
                while (curr_x >= end_x and curr_y >= end_y):
                    curr_x, curr_y = self.iterate_mask(
                        curr_x,
                        curr_y,
                        gradient,
                        major_dir_x,
                        dir_x,
                        dir_y
                    )
        return self.total_outside/self.total_checked

    def reorder(self, points_remaining, start, new, first):
        """
        Reorders the array to that of the shortest distanace between
        points given a certain starting point

        Arguments:
            points_remaining {2-D array} -- Array of the points yet to be
                                         sorted
            start {1-D array of shape (2)} -- Starting point for the sort
            new {None} -- Used in the recursion
            first {Bool} -- Used to determine if on first level of recursion

        Returns:
            int -- always 0
        """

        if first:
            distances = np.linalg.norm(points_remaining-start, axis=1)
            mask_cost = np.apply_along_axis(
                self.check_mask,
                1,
                points_remaining,
                start
            )
            distances = distances*mask_cost
            index = np.argmin(distances)
            new = np.array([[points_remaining[index], distances[index]]])
            self.reorder(
                np.delete(points_remaining, index, 0),
                points_remaining[index],
                new,
                False
            )
        elif len(points_remaining) > 1:
            distances = np.linalg.norm(points_remaining-start, axis=1)
            mask_cost = np.apply_along_axis(
                self.check_mask,
                1,
                points_remaining,
                start
            )
            distances = distances*mask_cost
            index = np.argmin(distances)
            new = np.append(
                new,
                [[points_remaining[index], distances[index]]],
                axis=0
            )
            self.reorder(
                np.delete(points_remaining, index, 0),
                points_remaining[index],
                new,
                False
            )
        else:
            new = np.append(
                new,
                [[
                    points_remaining[0],
                    np.linalg.norm(points_remaining-start)
                ]],
                axis=0
            )
            if self.total_distance > 0:
                values = new[:, 1]
                dist = np.sum(values)
                if dist < self.total_distance:
                    self.total_distance = dist
                    self.best = np.delete(new, 1, 1)
            else:
                values = new[:, 1]
                self.best = np.delete(new, 1, 1)
                self.total_distance = np.sum(values)
        return 0

    def iterate(self):
        """Used to interate through all possible lists to find shortest distance

            Returns:
                {(list[float],list[float])} - (x,y) points
        """

        for point in (self.points):
            self.reorder(self.points, point, None, True)
        x_values = []
        y_values = []
        log.debug("Path points: \n %s", self.points)
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        for best in self.best:
            x_values.append(best[0][0])
            y_values.append(best[0][1])
        return (x_values, y_values)


class Engine:
    """Engine class for image processing - Kmeans version
    """

    MOVECONST = 20

    def __init__(self):
        self.rope = rope.Rope()
        self.lace = None

    def nearestneighbours(self, plot, points, k):
        """Find the nearest point to another from a point map

        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the
                                   points
            points {tuple} -- (x,y) of the point to be found within the plot
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
            init='k-means++'
            ).fit(plot).cluster_centers_
        log.debug("Locations are: \n %s", locations)
        return locations

    def locate_y(self, arr):
        """Find the y value associated with nonzero value a

        Arguments:
            a {np.array} -- [x y] of the index of the needed y value within
                         self.lace

        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """

        y_location = self.lace[arr[0], arr[1]]
        return (arr[1], y_location)

    def locate_x(self, arr):
        """Find the x value associated with nonzero value a

        Arguments:
            a {np.array} -- [x y] of the index of the needed x value within
                         self.lace

        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """
        x_location = self.lace[arr[0], arr[1]]
        return (x_location, arr[0])

    def run(self, edges, mask):
        """ Used to run the processing on images

        Arguments:
            edges {np.array} -- Image in a greyscale format

        Returns:
            Rope -- full upadated rope obkect
        """

        self.lace = np.apply_along_axis(self.get_points, 0, edges)
        shoelace = np.nonzero(self.lace)  # Remove padded 0's
        combined_y = np.apply_along_axis(
            self.locate_y,
            1,
            np.transpose(shoelace)
        )
        self.lace = np.apply_along_axis(self.get_points, 1, edges)
        # * Transpose lace so that non-zero acts on the right axis.
        # * Transpose shoelace so that the points are outputs in array
        # * of [y x]
        shoelace = np.nonzero(self.lace)
        combined_x = np.apply_along_axis(
            self.locate_x,
            1,
            np.transpose(shoelace)
        )
        combined = np.concatenate((combined_x, combined_y))
        clusters = self.kmeans(combined[0::20])
        log.debug("clusters: \n %s", clusters)
        log.debug(np.shape(clusters))
        path = MaskPath(clusters, mask)
        x_locations, y_locations = path.iterate()
        log.debug("x: %s", x_locations)
        log.debug("y: %s", y_locations)
        for i in range(0, len(self.rope.lace)):
            self.rope.lace[i] = np.array([
                int(x_locations[i]),
                int(y_locations[i]),
                self.rope.lace[i][2]
                ])
        log.debug("rope: %s", self.rope.lace)
        return self.rope
