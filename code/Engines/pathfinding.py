import numpy as np
import sys
import logging as log
np.set_printoptions(threshold=sys.maxsize)



### TODO: TEST THIS
# class Node:
#     def __init__(self, parent=None,current, children ,distance=None):
#         self.parent = parent
#         self.current = current
#         self.children = children
#         self.distanace= distance
#     def newDistances(self):
#         self.distances=np.linalg.norm(np.delete(self.children,0,1)-current, axis=1)

#     def best(self):
#         bestDistance = np.inf
#         current = 0
#         while len(self.children)>current:
#             x=Node(self.current,self.children[current][0],np.delete(self.children,current,0), self.distanace+self.distances[current])
#             if x.distanace<bestDistance:
#                 self.bestChild = x
#             current +=1
#         if(len(self.bestChild.children))>0:
#             return self.bestChild.best()
#         else:
#             return self.bestChild.distanace


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
        return((x,y))
