import numpy as np
import cv2
import Rope
from scipy import spatial
import logging as log
import sys
np.set_printoptions(threshold=sys.maxsize)

class Video:
    """Class handling the processing on the images

        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or camera]
            out {string} -- [Output file for the video if being outputted]
    """

    MOVECONST = 10

    def __init__(self, cap, out):
        """Used to initialise the Video Class
        
        Arguments:
            cap {cv2.VideoCapture} -- [The video stream from the file or camera]
            out {string} -- [Output file for the video if being outputted]
        """

        log.info("Initialising Video Input")
        self.rope = Rope.Rope()
        self.cap = cap
        self.out = out
        log.debug("Finished Video Initialisation")


    def adjusted_mean(self,a):
        """Checks to make sure points are close enough together to be considered parts of a string
        
        Arguments:
            a {np.array} -- [2 points in an array to have mean found of]
        
        Returns:
            [float] -- [The mean of 2 points]
        """

        if a[1]-a[0]<100:
            return np.mean(a)
        else:
            return a[0]

    def get_points(self,a):
        """Returns the points of the shoelace by looking at the midpoints of 2 lines
        
        Arguments:
            a {np.array} -- 1-D array of all points on a given pixel line
        
        Returns:
            np.array -- Midpoints of all sets of 2 points
        """


        b = np.nonzero(a) # Get non-zero y values constituting edges
        z = b[0]
        log.debug("z length is: %s", len(z))
        if len(z)>1: # Make sure there are edges
            log.debug("Doing process")
            if len(z)%2!=0:
                if (z[1]-z[0])> (z[-2]-z[1]):
                    z =z[1:] 
                else:
                    z =z[0:-1]
            log.debug("z: %s", z)
            z=z.reshape(int(len(z)/2),2) #Reshape into points
            log.debug("reshaped: %s", z)
            if len(z)>1: ##incase only one edge exsists
                out=np.apply_along_axis(self.adjusted_mean,1,z)
                log.debug("mean: %s", out)
                out=np.apply_along_axis(np.round,0,out) #convert from flaot 
                log.debug("length z: %s length out: %s", len(z), len(out))
            else:
                out=[round(z.mean())]
            out = np.pad(out,(0,len(a)-len(out)),'constant',constant_values=(0,0)) #Output array same size as input array so numpy doesnt complain
            return out # Return the midpoints of each point
        else:
            return np.pad([0],(0,len(a)-len([0])),'constant',constant_values=(0,0)) #Zero Output

    def nearestneighbours(self,plot,points,k):
        """Find the nearest point to another from a point map
        
        Arguments:
            plot {2-D np.array} -- A collection of 2-D locations of all the points
            points {tuple} -- (x,y) of the point to be found within the plot
            k {int} -- The number of the closest point to be returned
        
        Returns:
            int -- index of the closest point within plot
        """

        tree = spatial.cKDTree(plot)
        _,indexes1 = tree.query(points, k=[k])
        return indexes1


    def locateY(self, a):
        """Find the y value associated with nonzero value a
        
        Arguments:
            a {np.array} -- [x y] of the index of the needed y value within self.lace
        
        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """

        y = self.lace[a[0], a[1]]
        return (a[1],y)
    def locateX(self, a):
        """Find the x value associated with nonzero value a
        
        Arguments:
            a {np.array} -- [x y] of the index of the needed x value within self.lace
        
        Returns:
            [tuple] -- (x,y) of the location within frame of the point
        """
        x = self.lace[a[0], a[1]]
        return (x,a[0])
    def shoelaceFinding(self):
        """Find the location of the shoelace

        Arguments:
            None
        """
        try:
            ret, frame = self.cap.read()
            width = frame.shape[0]
            log.debug("width: %s", width)
            log.debug("image resolution is: %s", frame.shape)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([220,220,220])
            upper_yellow = np.array([255,255,255])
            mask = cv2.inRange(frame, lower_yellow, upper_yellow)
            edges = cv2.Canny(mask, 50,60)
            self.lace = np.apply_along_axis(self.get_points, 0, edges)
            # log.debug("lace: \n %s", lace)
            shoelace = np.nonzero(self.lace) #Transpose lace so that non-zero acts on the right axis. Transpose shoelace so that the points are outputs in array of [y x]
            log.debug("shape shoelace: %s", np.shape(shoelace))
            combinedY = np.apply_along_axis(self.locateY, 1, np.transpose(shoelace))
            self.lace = np.apply_along_axis(self.get_points, 1, edges)
            shoelace = np.nonzero(self.lace) # * Transpose lace so that non-zero acts on the right axis. Transpose shoelace so that the points are outputs in array of [y x]
            combinedX = np.apply_along_axis(self.locateX, 1, np.transpose(shoelace))
            combined = np.concatenate((combinedX,combinedY))
            combinedReduced = combined[0::5]
            log.debug("combinedReduced points: \n %s", combinedReduced)
            log.info("number of points: %s", len(combinedReduced))
            map = np.zeros(len(combinedReduced))
            for i in range(0, self.rope.NO_NODES): #match each node to new point
                for k in range(1,self.rope.NO_NODES+1): #check each node isnt conflicting with another
                    log.debug("node: %s", i)
                    potentials = self.nearestneighbours(combinedReduced, (self.rope.lace[i][0],self.rope.lace[i][1]), k) #Find nearest matches to the point on camera from string model
                    if np.linalg.norm(self.rope.lace[i]-np.array([int(combinedReduced[potentials[0]][0]),int(combinedReduced[potentials[0]][1]),self.rope.lace[i][2]]))>self.MOVECONST:
                        if map[potentials[0]] == 0:
                            map[potentials[0]] = 1
                            self.rope.lace[i] = np.array([int(combinedReduced[potentials[0]][0]),int(combinedReduced[potentials[0]][1]),self.rope.lace[i][2]])
                            log.debug("potentials: %s", potentials)
                            log.debug("x,y: %s, %s", int(combinedReduced[potentials[0]][0]),int(combinedReduced[potentials[0]][1]))
                            log.debug("Moved: %s", k)
                            break
                    log.debug("Conflict: %s", k)
                    self.rope.lace[i] = np.array([int(combinedReduced[potentials[0]][0]),int(combinedReduced[potentials[0]][1]),self.rope.lace[i][2]])
            frame = self.rope.draw_point(frame)
            # cv2.imshow('edges', edges)
            # cv2.imshow('frame', frame)
            if self.out:
                self.out.write(frame)
                log.info("Writing Frame")
            log.debug("Rope positions, %s", self.rope.lace)
        except Exception as e:
            log.error("Exception occurred", exc_info=True)
            exit()
