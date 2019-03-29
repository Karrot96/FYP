import numpy as np
import sys
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=sys.maxsize)


class Paths:
    points = np.random.rand(100,2)
    totalDistance=-1
    best = None
    # def __init__(self):
    def plot(self):
        x=[]
        y=[]
        for i in range(0,len(self.points)):
            x.append(self.best[i,0][0])
            y.append(self.best[i,0][1])
        plt.plot(x,y)
        plt.show()

    def reorder(self, pointsRemaining, start, new, first):
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
        for i in range(0,len(self.points)):
            self.reorder(self.points, self.points[0],None, True)
            self.points = np.roll(self.points,1)
        return(self.best)


p = Paths()
for i in range(0,len(p.points)):
    print(i)
    p.reorder(p.points, p.points[0],None, True)
    p.points = np.roll(p.points,1)
    # p.plot()
p.plot()