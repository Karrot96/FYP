""" TODO: Currently Untested 15/5/19

Returns:
    [type] -- [description]
"""

import sys
import logging as log
import numpy as np
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


# ! This algorithm doesnt work. Needs more thought.
class LacePaths:
    """ Used to hande the Paths and search space

        Arguments:
            point {np.array} -- x,y coordinates of the point at end of path
            toSearch {np.array} -- The search space of remaining points
            path {list} -- List of the points in the current path

        Keyword Arguments:
            distance {int} -- The distance of the current path (default: {0})
        """
    def __init__(self, point, to_search, path, distance=0):
        """ Used to hande the Paths and search space

        Arguments:
            point {np.array} -- x,y coordinates of the point at end of path
            toSearch {np.array} -- The search space of remaining points
            path {list} -- List of the points in the current path

        Keyword Arguments:
            distance {int} -- The distance of the current path (default: {0})
        """

        self.end = point
        log.debug("Self.end: %s", self.end)
        self.to_search = to_search
        log.debug("Self.toSearch: %s", self.to_search)
        self.path = path
        log.debug("Self.path: %s", self.path)
        self.distance = distance
        log.debug("Self.distance: %s", self.distance)

    def next_search(self, i):
        """ Gets the new to search groups and removes the point from
        current search space

        Arguments:
            i {int} -- Index of the point to remove from the list

        Returns:
            np.array,bool -- tosearch array and if it is empty or not
        """
        self.to_search = np.delete(self.to_search, i, 0)
        if self.to_search:
            return self.to_search, True
        return self.to_search, False

    def finished(self, nodes):
        """ Works out if a complete path has been found

        Arguments:
            nodes {int} -- len of the number of points

        Returns:
            Bool -- Has the search been completed
        """
        if len(self.path) == nodes:
            return True
        else:
            return False

    def flip(self):
        """ Flips the path

        Returns:
            list -- flipped list of path
        """

        path = self.path
        path.reverse()
        return path


# TODO: remove laces with no search space
class BottomUp:
    """Used to do bottom up search with an adjusted A* based search

    Arguments:
        points {np.array} -- list of points in the search space
    """
    laces = []

    def __init__(self, points):
        """Used to do bottom up search with an adjusted A* based search

        Arguments:
            points {np.array} -- list of points in the search space
        """

        self.points = points
        for i, point in enumerate(self.points):
            # Current point at end of string, points to search, string
            self.laces.append(
                LacePaths(
                    point,
                    np.delete(self.points, i, 0),
                    [point]
                )
            )
        log.debug("Finished Init")

    def distance(self, start, points, index):
        """ Find the shortest distance to add from a search space

        Arguments:
            start {np.array} -- point to find distance from
            points {np.array} -- points to search
            index {int} -- index of the point in laces

        Returns:
            (int, float) -- tuple with the index and value of the shortest
                         distance
        """

        # get distances from point
        distances = np.linalg.norm(points - start, axis=1)
        new_distance = np.min(distances)+self.laces[index].distance
        return np.argmin(distances), new_distance

    def expand_search(self):
        """ Increase search space using the path with the shortest heuristic

        Returns:
            Bool -- Has it completed its search
        """
        # Find next best point by working out total distance looked at
        next_index, min_distance, point_index = self.next()
        point = self.laces[next_index].to_search[point_index]
        new_search = self.laces[next_index].next_search(point_index)
        path = self.laces[next_index].path
        path.append(np.array(point))
        log.debug("Path: %s", path)
        self.laces.append(
            LacePaths(
                point,  # New point
                new_search,  # updated search space
                path,  # updated lace
                distance=min_distance  # Min distance
            )
        )
        if self.laces[-1].finished(len(self.points)):
            return True
        else:
            new_path = self.laces[-1].flip()
            self.laces.append(
                LacePaths(
                    new_path[0],  # Inverse of above
                    new_search,
                    new_path,
                    distance=min_distance
                )
            )
            return False

    def next(self):
        """ finds the next path to expand with smallest heuristic

        Returns:
            (int, float, int) -- index of the best lace, value of the
            heuristic distance, index of point that produces that
        """

        curr_min = np.inf
        for i, lace in enumerate(self.laces):
            log.debug("next Searching: %s", lace.to_search)
            tmp_nxt, tmp_min = self.distance(
                lace.end,
                lace.toSearch,
                i
            )
            if tmp_min < curr_min:
                curr_min = tmp_min
                curr_min_index = tmp_nxt
                curr_nxt = i
        return curr_nxt, curr_min, curr_min_index

    def get_best(self):
        """Get the return value in correct format
        Returns:
            (list,list) -- tuple of x,y listss
        """

        x_values = []
        y_values = []
        # TODO This should be able to be done using np.apply_along_axis for
        # speed up
        for i, path in enumerate(self.laces[-1].path):
            x_values.append(path[0])
            y_values.append(path[1])
        return (x_values, y_values)

    def run(self):
        """ runs the class search

        Returns:
            (list,list) -- tuple of x,y listss
        """
        count = 0
        finished = False
        while not finished:
            finished = self.expand_search()
            count += 1
            log.debug("Count: %s", count)
        log.info("Count: %s", count)
        return self.get_best()


class ShortestPaths:
    """Used to hande the finding of the shortest path to connectg a set of
    points - definitely not efficient

        Variabels:
            points {2-D numpy array} - list of points to be connected
    """
    totalDistance = -1
    best = None

    def __init__(self, points):
        """Used to hande the finding of the shortest path to connectg a set
        of points - definitely not efficient
        Variabels:
            points {2-D numpy array} - list of points to be connected
        """
        self.points = np.insert(points, 0, np.arange(len(points)), axis=1)


class Paths:
    """Used to hande the finding of the shortest path to connectg a set of
    points - definitely not efficient

        Variabels:
            points {2-D numpy array} - list of points to be connected
    """
    total_distance = -1
    best = None

    def __init__(self, points):
        """Used to hande the finding of the shortest path to connectg a
        set of points - definitely not efficient

        Variabels:
            points {2-D numpy array} - list of points to be connected
        """
        self.points = points
        log.debug("Points in path: \n %s", self.points)

    # ! work out wasted processing
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
