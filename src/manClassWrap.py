# Class for multiprocessing Astar. Used for manhattan heuristic only as it runs slow.
import sys
import math
import numpy as np 
import copy 
from simulator.sokoban_world import SokobanWorld
import settings
import argparse
import utils
import os
import console_utils
import matplotlib
import matplotlib.pyplot as plt
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop

#from chebnetskip import GCN


class SokoNode:

    def __init__(self, sokoenv):
        self.nEnv = sokoenv
        self.obs = self.nEnv.get_observation()
        self.boxPos = np.nonzero(self.obs[:,:,2])
        #self.action = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    def neighbors(self):
        nbList = []
        envList = []
        rpos = []
        for i in range(4):
            envList.append(copy.deepcopy(self.nEnv))
            action_onehot = np.zeros([5], dtype='bool')
            action_onehot[i] = 1
            rpos.append(envList[i]._robot_loc)
            envList[i].step(action_onehot)
            if not np.array_equal(envList[i]._robot_loc,rpos[i]):
                #print("step is true")
                nbList.append(SokoNode(envList[i]))
        #print("neighborlist")
        #print(len(nbList))
        return nbList
        
    # Return observation
    def observation(self):
        return self.obs

Infinite = float('inf')

class AStar:
    __metaclass__ = ABCMeta
    __slots__ = ()

    class SearchNode:
        __slots__ = ('data', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None
            

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v
    
    @abstractmethod
    def heuristic_cost_estimate(self, current, goal):
        """Computes the estimated (rough) distance between a node and the goal, this method must be implemented in a subclass. The second parameter is always the goal."""
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1, n2):
        """Gives the real distance between two adjacent nodes n1 and n2 (i.e n2 belongs to the list of n1's neighbors).
           n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
           This method must be implemented in a subclass."""
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node):
        """For a given node, returns (or yields) the list of its neighbors. this method must be implemented in a subclass"""
        raise NotImplementedError

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def isNodeClosed(self, node, obslist):
        for closed in obslist:
           if np.array_equal(node.data.obs,closed):
              node.closed = True

    def astar(self, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        nodesExplored = 0
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        closedSet = []
        heappush(openSet, startNode)
        while openSet:
            nodesExplored += 1
            current = heappop(openSet)
            if nodesExplored == 100000:
                return None
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            closedSet.append(current.data.obs)
            for neighbor in [searchNodes[n] for n in self.neighbors(current.data)]:
                #if not current.came_from == None:
                #    a = current.came_from.data.obs
                #    b = neighbor.data.obs
                #    if np.array_equal(a, b):
                #        neighbor.closed = True
                self.isNodeClosed(neighbor,closedSet)
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
        return None


def find_path(start, goal, neighbors_fnct, reversePath=False, heuristic_cost_estimate_fnct=lambda a, b: Infinite, distance_between_fnct=lambda a, b: 1.0, is_goal_reached_fnct=lambda a, b: a == b):
    """A non-class version of the path finding algorithm"""
    class FindPath(AStar):

        def heuristic_cost_estimate(self, current, goal):
            return heuristic_cost_estimate_fnct(current, goal)

        def distance_between(self, n1, n2):
            return distance_between_fnct(n1, n2)

        def neighbors(self, node):
            return neighbors_fnct(node)

        def is_goal_reached(self, current, goal):
            return is_goal_reached_fnct(current, goal)
    return FindPath().astar(start, goal, reversePath)


class SokoSolver(AStar):

    def __init__(self, h, g):
        self.nExplored = 0
        self.h = h
        #if self.h == "NN":
            #self.net = SokoNet()
            #self.net = GCN()
            #self.net.load_state_dict(torch.load("gcncheb_med2_1e4_64_2_09t_005vt.pth"))
            #self.net.to("cuda:0")
            #self.net.eval()
            #self.g = g
        #elif self.h == "GCN":
            #self.net = SokoNet()
            #self.net = GCN()
            #self.net.load_state_dict(torch.load("gcncheb4skipcontest.pth"))
            #self.net.to("cpu")
            #self.net.eval()
       	    #self.g = self.createGraph(g).to("cpu")

            
    

    def heuristic_cost_estimate(self, node, goal):

        if self.h == "manhattan":
            #print(np.abs(node.nEnv._robot_loc[0]-node.boxPos[0][0])+np.abs(node.nEnv._robot_loc[1]-node.boxPos[1][0])+np.abs(node.boxPos[0][0]-goal.boxPos[0][0])+np.abs(node.boxPos[1][0]-goal.boxPos[1][0]))
            #return np.abs(node.nEnv._robot_loc[0]-node.boxPos[0][0])+np.abs(node.nEnv._robot_loc[1]-node.boxPos[1][0])+np.abs(node.boxPos[0][0]-goal.boxPos[0][0])+np.abs(node.boxPos[1][0]-goal.boxPos[1][0])
            return np.abs(node.boxPos[0][0]-goal.boxPos[0][0])+np.abs(node.boxPos[1][0]-goal.boxPos[1][0])
        elif self.h == "NN":
            # neural network here
            #net_input = [np.concatenate((node.obs,goal.obs),axis=2)]
            #tb_input = torch.from_numpy(np.array(net_input,dtype="float32")).to("cuda:0")
            #tb_input = tb_input.permute(0,3,1,2)
            #return self.net(tb_input).item()
            return 1
        elif self.h == "GCN":
            # neural network here
            #tb_input = torch.from_numpy(np.array(np.concatenate((node.obs,goal.obs),axis=2).reshape(X**2,8),dtype="Float32")).to("cpu")
            #print(tb_input.size())
            #return self.net(tb_input,self.g).item()
            return 1
        else:
            # BFS
            return 1

    def distance_between(self, n1, n2):
        return 1

    def neighbors(self, node):
        #print("get neighbors")
        nb = node.neighbors()
        
        return node.neighbors()

    def is_goal_reached(self, node, goal):   
        self.nExplored += 1 
        self.nEnv = None
        nobs = node.obs
        gobs = goal.obs
        nobs = nobs[:,:,2]
        gobs = gobs[:,:,2]
        #print(nobs)
        #print(gobs)
        ret = np.array_equal(nobs,gobs)
        #print(ret)
        return ret

    def createGraph(self, imglen):

        #g = graph
        #g.add_nodes(imglen**2)
        src = []
        dst = []
        for i in range(imglen):
            for k in range(imglen):
                if (k < imglen-1):
                    src.append(imglen*i+k)
                    dst.append(imglen*i+k+1)
                if (i < imglen-1):
                    src.append(imglen*i+k)
                    dst.append(imglen*i+k+imglen)

        #g.add_edges(src,dst)
        #g.add_edges(dst,src)

        outsrc = np.concatenate((src,dst))
        outdst = np.concatenate((dst,src))
        #print(len(outsrc))
        # show the network
        # import networkx as nx
        # import matplotlib.pyplot as plt
        # fig = plt.figure(dpi=150)
        # nx_g = g.to_networkx().to_undirected()
        # pos = nx.spectral_layout(nx_g)
        # nx.draw(nx_g, pos, with_labels=True, node_color=[[.7,.7,.7]])
        # plt.show()

        return torch.tensor([outsrc,outdst])

def solveWrapper(tupNode):

    sokopathM = SokoSolver("manhattan",0)

    spathM = sokopathM.astar(tupNode[0], tupNode[1])
    return (sokopathM.nExplored, len(list(spathM)))



#ind = np.arange(len(xLevelSize))
#xt = ["9","12","15","18"]
# Plot everything
# Plot solution length
#fig1, ax1 = plt.subplots()

#ax1.plot(xLevelSize, mPlotPlan, label="A* M")
#ax1.plot(xLevelSize, nnPlotPlan, label="A* NN")

#ax1.set_title('Solution Length Per Level Size')
#ax1.set_ylabel('Solution Length')
#ax1.set_xlabel('Level Size')
#ax1.set_xticks(xLevelSize)
#ax1.set_xticklabels(xt)

#ax1.legend()

#fig1.tight_layout()
#fig1.savefig(filenameout+"Plan.png")
#fig1.savefig(filenameout+"Plan.svg")


# Plot nodes explored
#fig2, ax2 = plt.subplots()

#ax2.plot(xLevelSize, mPlotExplored, label="A* M")
#ax2.plot(xLevelSize, nnPlotExplored, label="A* NN")

#ax2.set_title('Nodes Explored Per Level Size')
#ax2.set_ylabel('Avg. Explored Nodes')
#ax2.set_yscale('log')
#ax2.set_xlabel('Level Size')
#ax2.set_xticks(xLevelSize)
#ax2.set_xticklabels([str(x) for x in xLevelSize])

#ax2.legend()

#fig2.tight_layout()
#fig2.savefig(filenameout+"Explored.png")
#fig2.savefig(filenameout+"Explored.svg")


# Plot success rate

#fig, ax = plt.subplots()
#barwidth = 0.35

#ind = np.arange(len(xLevelSize))

#ax.bar(ind-barwidth/2, mPlotSuccess, barwidth, label="A* M")
#ax.bar(ind+barwidth/2, nnPlotSuccess, barwidth, label="A* NN")

#ax.set_title('Success Rate Per Level Size')
#ax.set_ylabel('Success Rate')
#ax.set_xlabel('Level Size')
#ax.set_xticks(ind)
#ax.set_xticklabels([str(x) for x in xLevelSize])
#ax.legend()

#fig.tight_layout()
#fig.savefig(filenameout+"Success.png")
#fig.savefig(filenameout+"Success.svg")

#print(mPlotSuccess)
#print(nnPlotSuccess)

#print(
   
# finally:
#     filepath = f.filename
#     f.close()
#     if filepath.startswith(TEMP_DIR):
#         os.remove(filepath)
