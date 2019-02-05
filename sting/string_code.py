import pyglet
from pyglet.gl import *
from pyglet.window import key
from OpenGL.GLUT import *

import math
import numpy as np
import csv


FRAME_RATE = 6
meh = 100
lol = 0
class Rope:
    NO_NODES = 100
    DISTANCE_BETWEEN_NODES = 10
    lace = np.zeros((NO_NODES,3))

    def __init__(self):
        for i in range(0,len(self.lace)):
            self.lace[i] = np.array([i*self.DISTANCE_BETWEEN_NODES, 50 ,0])

    def Follow_The_Leader(self, move, node):
        newVector = node - (move+([0,0.5*9.81*(1/FRAME_RATE),0]))
        print(newVector)
        # newVector = node - move
        # print(newVector)
        movementVector = newVector/np.absolute(newVector[np.argmax(np.absolute(newVector))])
        # print(movementVector)
        move = move+movementVector
        x = math.sqrt((move[0]-node[0])**2+(move[1]-node[1])**2+(move[2]-node[2])**2)
        print(x)
        while x>self.DISTANCE_BETWEEN_NODES:
            move = move+movementVector
            x= math.sqrt((move[0]-node[0])**2+(move[1]-node[1])**2+(move[2]-node[2])**2)
        # print("end")
        # print(move)
        return move
    # This should be done recursively would be more efficient
    def Implement_Follow_The_Leader(self, node, position):
        originalNode = node
        self.lace[node] = position
        while node>0:
            currentNode = node-1
            self.lace[currentNode] = self.Follow_The_Leader(self.lace[currentNode],self.lace[node])
            node = currentNode
        node = originalNode
        while node<len(self.lace)-1:
            currentNode = node+1
            self.lace[currentNode] = self.Follow_The_Leader(self.lace[currentNode],self.lace[node])
            node = currentNode

    #nodes in form np.array([[[NodeID],[x,y,z]], [[NodeID],[x,y,z]]])
    def forced_nodes(nodes):
        np.sort(nodes, axis=0)
        if len(nodes) ==1:
            Implement_Follow_The_Leader(nodes[0][0][0], nodes[0][1])
        elif len(nodes) >= 1:


rope = Rope()

window = pyglet.window.Window(1000,1000)

def write_csv(lace):
    with open('rope_5.csv', mode='w') as rope_file:
        rope_file = csv.writer(rope_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in lace:
            rope_file.writerow(i)

def PointsInCircum(r, n=25, pi=3.14):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
pts = np.array(PointsInCircum(20))

# function that increments to the next
# point along a circle
frame = 0
def update_frame(x, y):
    global frame
    if frame == None or frame == pts.shape[0]-1:
        frame = 0
    else:
        frame += 1

print(rope.lace[80])

count = 0

@window.event
def on_key_press(symbol, modifiers):
    global y
    global count
    count+=1
    try:
        y = y+5
    except:
        y=100
    rope.Implement_Follow_The_Leader(0, [5, y,0])
    rope.Implement_Follow_The_Leader(99, [990, y,0])
    # print(y)
    # print(rope.lace)
    if count == 10:
        print("here")
        write_csv(rope.lace)
    on_draw()

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_LINE_STRIP)
    # create a line, x,y,z
    for i in rope.lace:
        # print(i)
        glVertex3f(round(i[0]), round(i[1]), round(i[2]))
    glEnd()

pyglet.app.run()
