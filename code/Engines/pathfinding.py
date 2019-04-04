import numpy as np
import sys
import logging as log
np.set_printoptions(threshold=sys.maxsize)

class LacePaths:
    """ Used to hande the Paths and search space
        
        Arguments:
            point {np.array} -- x,y coordinates of the point at end of path
            toSearch {np.array} -- The search space of remaining points
            path {list} -- List of the points in the current path
        
        Keyword Arguments:
            distance {int} -- The distance of the current path (default: {0})
        """
    def __init__(self, point, toSearch, path,distance=0):
        """ Used to hande the Paths and search space
        
        Arguments:
            point {np.array} -- x,y coordinates of the point at end of path
            toSearch {np.array} -- The search space of remaining points
            path {list} -- List of the points in the current path
        
        Keyword Arguments:
            distance {int} -- The distance of the current path (default: {0})
        """

        self.end=point
        self.toSearch = toSearch
        self.path = path
        self.distanace = distance
    def nextSearch(self,i):
        """ Gets the new to search groups and removes the point from current search space
        
        Arguments:
            i {int} -- Index of the point to remove from the list
        
        
        Returns:
            np.array -- tosearch array
        """
        self.toSearch=np.delete(self.toSearch,i,0)
        return np.delete(self.toSearch,i,0)
    def finished(self, nodes):
        """ Works out if a complete path has been found
        
        Arguments:
            nodes {int} -- len of the number of points
        
        Returns:
            Bool -- Has the search been completed
        """

        if len(self.path)==nodes:
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
        for i in points:
            self.laces.append(LacePaths(i,np.delete(points,i,0),[i])) # Current point at end of string, points to search, string

    def distance(self, start, points, index):
        """ Find the shortest distance to add from a search space
        
        Arguments:
            start {np.array} -- point to find distance from
            points {np.array} -- points to search
            index {int} -- index of the point in laces
        
        Returns:
            (int, float) -- tuple with the index and value of the shortest distance
        """

        distances = np.linalg.norm(points - start, axis=1) # get distances from point
        return np.argmin(distances), np.min(distances)+self.laces[index].distance

    def expandSearch(self):
        """ Increase search space using the path with the shortest heuristic
        
        Returns:
            Bool -- Has it completed its search
        """

        nextIndex, minDistance, pointIndex = self.next() #Find next best point by working out total distance looked at
        newSearch = self.laces[nextIndex].nextSearch(pointIndex)
        self.laces.append(
            LacePaths(
                self.laces[nextIndex].toSearch[pointIndex], #New point
                newSearch,  #updated search space
                self.laces[nextIndex].path.append(self.laces[nextIndex].toSearch[pointIndex]), # updated lace
                distance=minDistance # Min distance
            )
        )
        if self.laces[-1].finished(len(self.points)):
            return True
        else:
            newPath = self.laces[-1].flip()
            self.laces.append(
                LacePaths(
                    newPath[0], #Inverse of above
                    newSearch,
                    newPath,
                    distance=minDistance
                )
            )
            return False

    def next(self):
        """ finds the next path to expand with smallest heuristic
        
        Returns:
            (int, float, int) -- index of the best lace, value of the heuristic distance, index of point that produces that
        """

        currMin = np.inf
        for i in range(len(self.laces)):
            tmpNext,tmpMin=self.distance(self.laces[i].end,self.laces[i].toSearch,i)
            if tmpMin<currMin:
                currMin = tmpMin
                currMinIndex = tmpNext
                currNext=i
        return currNext, currMin, currMinIndex
    def getBest(self):
        """Get the return value in correct format
        
        Returns:
            (list,list) -- tuple of x,y listss
        """

        x=[]
        y=[]
        for i in range(len(self.laces[-1].path)):
            x.append(self.laces[-1].path[i][0])
            y.append(self.laces[-1].path[i][1])
        return (x,y)
    def run(self):
        """ runs the class search
        
        Returns:
            (list,list) -- tuple of x,y listss
        """

        finished = False
        while not finished:
            self.expandSearch()
        return self.getBest()
        

class ShortestPaths:
    """Used to hande the finding of the shortest path to connectg a set of points - definitely not efficient
    
        Variabels:
            points {2-D numpy array} - list of points to be connected
    """
    totalDistance=-1
    best = None
    def __init__(self, points):
        """Used to hande the finding of the shortest path to connectg a set of points - definitely not efficient
    
        Variabels:
            points {2-D numpy array} - list of points to be connected
        """
        self.points = np.insert(points,0,np.arange(len(points)),axis=1)

    


class Paths:
    """Used to hande the finding of the shortest path to connectg a set of points - definitely not efficient
    
        Variabels:
            points {2-D numpy array} - list of points to be connected
    """
    totalDistance=-1
    best = None
    def __init__(self, points):
        """Used to hande the finding of the shortest path to connectg a set of points - definitely not efficient
    
        Variabels:
            points {2-D numpy array} - list of points to be connected
        """
        self.points = points
        log.debug("Points in path: \n %s", self.points)

    # ! work out wasted processing
    def reorder(self, pointsRemaining, start, new, first):
        """ Reorders the array to that of the shortest distanace between points given a certain starting point

        Arguments:
            pointsRemaining {2-D array} -- Array of the points yet to be sorted
            start {1-D array of shape (2)} -- Starting point for the sort
            new {None} -- Used in the recursion
            first {Bool} -- Used to determine if on first level of recursion

        Returns:
            int -- always 0
        """

        if first:
            distances = np.linalg.norm(pointsRemaining-start, axis=1)
            x=np.argmin(distances)
            new= np.array([[pointsRemaining[x],distances[x]]])
            self.reorder(np.delete(pointsRemaining,x,0),pointsRemaining[x],new,False)
        elif len(pointsRemaining)>1:
            distances = np.linalg.norm(pointsRemaining-start, axis=1)
            x=np.argmin(distances)
            new=np.append(new,[[pointsRemaining[x],distances[x]]],axis=0)
            self.reorder(np.delete(pointsRemaining,x,0),pointsRemaining[x],new, False)
        else:
            new=np.append(new,[[pointsRemaining[0], np.linalg.norm(pointsRemaining-start)]],axis=0)
            if self.totalDistance>0:
                values = new[:,1]
                dist = np.sum(values)
                if dist<self.totalDistance:
                    self.totalDistance=dist
                    self.best = np.delete(new,1,1)
            else:
                values = new[:,1]
                self.best = np.delete(new,1,1)
                self.totalDistance=np.sum(values)
        return 0
    def iterate(self):
        """Used to interate through all possible lists to find shortest distance

            Returns:
                {(list[float],list[float])} - (x,y) points
        """

        for i in range(0,len(self.points)):
            self.reorder(self.points, self.points[0],None, True)
            self.points = np.roll(self.points,1,axis=0)
        x=[]
        y=[]
        log.debug("Path points: \n %s", self.points)
        for i in range(0,len(self.points)):
            x.append(self.best[i,0][0])
            y.append(self.best[i,0][1])
        return (x,y)
