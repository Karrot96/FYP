import numpy as np
import sys
import logging as log
np.set_printoptions(threshold=sys.maxsize)



class Path:
    """ Path used to hold the node of the search tree. Each node contains a
            head which signifies the most recent addition to the path. A search list
            containing nodes not in the path. A list of nodes that have been added below. 
            And the full path to this point.
            A total distance to this point 
        
        Arguments:
            head {np.array } -- [description]
            to_search {np.array} -- [description]
            path {np.array} -- [description]
        """
    # Head of the path
    # nodes not in the path
    # nodes already searched in the list to search
    # the full connected path
    def __init__(self, head, to_search, path,distance=0):
        """ Path used to hold the node of the search tree. Each node contains a
            head which signifies the most recent addition to the path. A search list
            containing nodes not in the path. A list of nodes that have been added below. 
            And the full path to this point. 
        
        Arguments:
            head {np.array } -- [description]
            to_search {np.array} -- [description]
            path {np.array} -- [description]
        """
        self.distance = distance
        self.head = head
        self.to_search = to_search
        self.search_distances = np.linalg.norm(to_search - self.head, axis=1)
        self.searched=np.zeros(len(to_search))
        self.path = path
    def shortest(self):
        self.shortest_index = np.argmin(self.search_distances)
        return self.distance+self.search_distances[self.shortest_index]
    def selected(self):
        distance = self.distance+self.search_distances[self.shortest_index]
        point = self.to_search[self.shortest_index]
        new_to_search = np.delete(self.to_search,self.shortest_index,0)
        self.search_distances[self.shortest_index] = np.inf
        self.searched[self.shortest_index] = 1 
        finished_search_of_node = True
        for i in self.searched:
            if i == 0:
                finished_search_of_node=False
        return distance, point, new_to_search, self.path.append(point), finished_search_of_node
    
class AStar:
    laces = []
    def __init__(self, points):
        log.debug("Starting AStar graph expansion")
        self.original_points = points
        for i in range(len(self.original_points)):
            self.laces.append(
                Path(
                    self.original_points[i],
                    np.delete(self.original_points,i,0),
                    self.original_points[i]
                )
            )
        log.debug("Initialisation completed for A* graph")

    def expand_search(self):
        curr_min = np.inf
        for i in range(len(self.laces)):
            log.debug("Currently searching node %s of laces.",i)
            tmp_min = self.laces[i].shortest()
            if tmp_min<curr_min:
                curr_min=tmp_min
                best_lace_index = i
        new_distance,point,to_search,path,remove_node=self.laces[best_lace_index].selected()
        if len(self.original_points)==len(path):
            log.debug("Optimal path is: %s ", path)
            self.path = path
            return True
        if remove_node:
            np.delete(self.laces,best_lace_index,0)
        self.laces.append(
            Path(
                point,
                to_search,
                path,
                distance = new_distance
            )
        )
        new_path = self.laces[-1].path.flip()
        self.laces.append(
            Path(
                new_path[0],
                to_search,
                new_path,
                distance = new_distance
            )
        )

    def get_best(self):
        x=[]
        y=[]
        for i in range(len(self.path)):
            x.append(self.path[i][0])
            y.append(self.path[i][1])
        return (x,y)
            
    def run(self):
        count = 0
        finished = False
        while not finished:
            finished=self.expand_search()
            count+=1
            log.debug("Count: %s",count)
        log.info("Count: %s",count)
        return self.get_best()
        # Bottom up approach. Start with each node as a possible
        # starting point. Expand the search tree to the path with the lowest
        # total distance when adding a new node.



