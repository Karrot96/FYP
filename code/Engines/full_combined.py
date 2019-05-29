import sys
import logging as log
import numpy as np
import rope
from sklearn.cluster import KMeans
from scipy import spatial
from scipy.optimize import linear_sum_assignment
np.set_printoptions(threshold=sys.maxsize)

MOVE_THRESH = 20


class ShortestPath:

    def __init__(self, points):
        self.points = points
        self.total_distance = -1
        self.best = None
        log.debug("Points in ShortestPath Class: \n %s", self.points)

    def reorder(self, points_remaining, start, new, first):
        """ Reorders the array to that of the shortest distanace between
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
            index = np.argmin(distances)
            new = np.array([[points_remaining[index], distances[index]]])
            self.reorder(
                np.delete(points_remaining, index, 0),
                points_remaining[index],
                new,
                False
            )
        elif len(points_remaining) > 1:
            log.debug(" %s: Number of points remaining \
                      in ShortestPath reorder: %s",
                      __name__,
                      len(points_remaining)
            )
            distances = np.linalg.norm(points_remaining-start, axis=1)
            index = np.argmin(distances)
            new = np.append(
                new,
                [[
                    points_remaining[index],
                    distances[index]
                ]],
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

        for point in self.points:
            self.reorder(self.points, point, None, True)
        x_value = []
        y_value = []
        log.debug("Path points: \n %s", self.points)
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        for best in self.best:
            x_value.append(best[0][0])
            y_value.append(best[0][1])
        return (x_value, y_value)


class Points:

    def __init__(self):
        self.points = None
        self.lace = None

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

    def get_points(self, arr):
        """Returns the points of the shoelace by looking for edges

        Arguments:
            a {np.array} -- 1-D array of all points on a given pixel line

        Returns:
            np.array -- all non-zero points on a line
        """
        non_zero = np.nonzero(arr)  # Get non-zero y values constituting edges
        first_non_zero = non_zero[0]
        # if first_non_zero.any():
        #     log.debug("A:%s B:%s", len(arr), len(first_non_zero))
        #     log.debug("Z:%s", first_non_zero)
        # Output array same size as input array so numpy doesnt complain
        out = np.pad(
            first_non_zero,
            (0, len(arr)-len(first_non_zero)),
            'constant',
            constant_values=(0, 0)
            )
        return out  # Return the midpoints of each point

    def connected_dots(self, edges):
        self.lace = np.apply_along_axis(self.get_points, 0, edges)
        shoelace = np.nonzero(self.lace)  # Remove padded 0's
        combined_y = np.apply_along_axis(
            self.locate_y,
            1,
            np.transpose(shoelace)
            )
        self.lace = np.apply_along_axis(self.get_points, 1, edges)
        # * Transpose lace so that non-zero acts on the right axis.
        # * Transpose shoelace so that the points are outputs in array of [y x]
        shoelace = np.nonzero(self.lace)
        combined_x = np.apply_along_axis(
            self.locate_x,
            1,
            np.transpose(shoelace)
            )
        combined = np.concatenate((combined_x, combined_y))
        self.points = combined


class Engine:

    def __init__(self):
        """Initialisation of the engine
        first is used for finding the most likely positioning of the rope on
        the first run through.
        A different more exhaustive and slower approach is taken for this
        search
        """
        self.rope = rope.Rope()
        self.first = True
        self.lace = None

    def nearestneighbours(self, plot, points, k):
        """Find the nearest point to another from a point map

        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned

        Returns:
            int -- index of the closest point within plot
        """

        tree = spatial.cKDTree(plot)
        distance, indexes1 = tree.query(points, k=[k])
        # log.info(distance)
        # log.info(indexes1)
        return indexes1, distance

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

    def run(self, edges):
        points = Points()
        # Method
        # TODO write and test more methods into class Points
        points.connected_dots(edges)
        # Global method
        kmeans_data = points.points
        search_points = self.kmeans(kmeans_data)
        log.debug("search_points: \n %s", search_points)
        if self.first:
            path = ShortestPath(search_points)
            x_locations, y_locations = path.iterate()
            # log.debug("x: %s", x_locations)
            # log.debug("y: %s", y_locations)
            self.rope.find_lace(x_locations, y_locations)
            self.first = False
        else:
            nodes = []
            # TODO Hungarian could be used to perform matching
            hungarian = True
            if hungarian:
                for i in self.rope.lace:
                    nodes.append(search_points-i)
                log.info("nodes space : %s", np.shape(nodes))
                row_ind, col_ind = linear_sum_assignment(nodes)
                for i in range(len(row_ind)-1):
                    begining = False
                    end = False
                    if i == 0:
                        begining = True
                    if i == len(row_ind)-2:
                        end = True
                    self.rope.implement_follow_the_leader(row_ind[i],
                                                          search_points[col_ind[i]],
                                                          begining,
                                                          end,
                                                          row_ind[i+1],
                                                          search_points[col_ind[i+1]]
                    )
                    
            else:
                for i in search_points:
                    search_space = np.array(self.rope.lace)
                    position, distanace = self.nearestneighbours(search_space[:, :2], i, 1)
                    if distanace > MOVE_THRESH:
                        nodes.append([position, np.append(i, -1)])
                nodes = np.array(nodes)
                log.debug(nodes)
                sorted_nodes = nodes[nodes[:,0].argsort()]
                length = len(sorted_nodes)-1
                for j in range(length):
                    i = length-j
                    log.debug(i)
                    log.debug(sorted_nodes[i][0])
                    if sorted_nodes[i][0][0] == sorted_nodes[i-1][0][0]:
                        log.debug(i)
                        sorted_nodes = np.delete(sorted_nodes, (i), axis=0)
                log.debug("Sorted_nodes: \n %s", sorted_nodes)
                for i in range(len(sorted_nodes)-1):
                    begining = False
                    end = False
                    if i == 0:
                        begining = True
                    if i == len(nodes)-2:
                        end = True
                    self.rope.implement_follow_the_leader(sorted_nodes[i][0][0],
                                                        sorted_nodes[i][1],
                                                        begining,
                                                        end,
                                                        sorted_nodes[i+1][0][0],
                                                        sorted_nodes[i+1][1]
                    )
            log.debug("Lace: \n %s", self.rope.lace)
        return self.rope
