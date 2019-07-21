#import astar
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
import manClassWrap
from manClassWrap import SokoNode
#from chebnetskip import GCN

if sys.version_info[0] == 2:
    input = raw_input


# Parse arguments
parser = argparse.ArgumentParser(description='Generate data.')
parser.add_argument('dataset_name', type=str,
                    help='Name of the dataset you want to view.')
parser.add_argument('-r', '--randomize', action='store_true',
                    help='Randomize all samples.')
parser.add_argument('-g', '--grouped', action='store_true',
                    help='Randomize instances only (keep variations grouped).')
parser.add_argument('-s', '--screenshot', action='store_true',
                    help='Enable screenshot mode.')
args = parser.parse_args()


# Define params
filenameout = "sokostamanhattan"


# Auto generated params:
TEMP_DIR = "/tmp"


if __name__ == '__main__':    

    level_list = ["medium9_2", "medium12_2", "medium15_2", "medium18_2"]
    levelsize = [11,14,17,20]
    reslist = []

    for idx, level in enumerate(level_list):

    # Load data and initialize grid world
    #f = utils.load_dataset(args.dataset_name, TEMP_DIR)
        f = utils.load_dataset(level, TEMP_DIR)
        X, Y = f.attrs["world_shape"]
        env = SokobanWorld()
    
        plotdict = {}
        plotdict["avgManhattanExplored"] = 0
        plotdict["avgCNNExplored"] = 0
        plotdict["testSize"] = 0
        plotdict["avgPlanLength"] = 0
        plotdict["avgManhattanPlanLength"] = 0
        plotdict["avgCNNPlanLength"] = 0
        plotdict["CNNTestSize"] = 0
        plotdict["ManhattanTestSize"] = 0

        try:
            # Print dataset info
            print ("")
            print ("Description: {}".format(f.attrs["description"]))
            print ("World Dims: {}".format(f.attrs["world_shape"]))
            print ("Number of Objects: {}".format(f.attrs["num_objects"]))
            print ("Number of Valid Actions: {}".format(f.attrs["num_actions"]))
            print ("")
    
            # Animate data set
            num_actions = f.attrs["num_actions"]
            indices = range(f.attrs["num_samples"])
            dim = f.attrs["world_shape"][0]


            # Train percentage split 
            t_p = 0.9
            train_split = t_p*f.attrs["num_samples"]
            test_split = train_split+f.attrs["num_samples"]*(1.0-t_p)/2
            train_index = np.linspace(0,int(train_split),int(train_split),dtype=int)
            test_index = indices[int(train_split):int(test_split)]
            val_index = indices[int(test_split):]

            #plotdict["testSize"] = len(test_index)
            testNodes = []
            #goalNodes = []
            for index,i in enumerate(test_index[:100]):
            
                T =  f["sequence_length"][i]
                print(T)
                if T < 35:
                    env.reset_world(f["wall"][i], f["robot_loc"][i,0], f["obj_loc"][i,:,0], f["goal_loc"][i])
                    plotdict["testSize"] += 1
                # [:,:,0] is wall layer, 
                # [:,:,1] is goal layer,
                # [:,:,2] is object layer, 
                # [:,:,3] is robot layer
                    init_img_state = env.get_observation()
                    goal_img_state = init_img_state.copy()
                    goal_img_state[:,:,3] *= False # Robot layer is zeroed out
                    goal_img_state[:,:,2] = goal_img_state[:,:,1] # Objects should be at goal
            
                    startNode = SokoNode(env)
                    goalNode = SokoNode(env)
                    goalNode.obs = goal_img_state.copy()
                    goalNode.boxPos = np.nonzero(goal_img_state[:,:,1])
                    testNodes.append((copy.deepcopy(startNode),copy.deepcopy(goalNode)))
                #startNodes.append(startNode)
                #goalNodes.append(goalNode)
                #print(startNode.boxPos)
                #print(goalNode.boxPos)
            pool = mp.Pool(processes=7)
       
            results = pool.map(manClassWrap.solveWrapper, testNodes)
                #spathM = None
                #print(levelsize[idx])
                #sokopathNN = SokoSolver("GCN", levelsize[idx])
                #spathNN = sokopathNN.astar(startNode, goalNode)
                #spathNN = None

        #plotdict["avgPlanLength"] += T
        #plotdict["testSize"] += 1
            for manTup in results:
                if not manTup[0] == 100000:
                    plotdict["avgManhattanPlanLength"] += manTup[1]
                    plotdict["avgManhattanExplored"] += manTup[0]
                    plotdict["ManhattanTestSize"] += 1
                     
            reslist.append(plotdict)
    #except IOError:
     #   print("oh no")
        finally:
            filepath = f.filename
            f.close()
            if filepath.startswith(TEMP_DIR):
                os.remove(filepath)
      
    



#fig, (ax1, ax2) = plt.subplots(2)
    mPlotPlan = []
    nnPlotPlan = []
    mPlotExplored = []
    nnPlotExplored = []
    mPlotSuccess = []
    nnPlotSuccess = []

    xLevelSize = [9,12,15,18]
    idx = 0
    for res in reslist:
        mPlotPlan.append(res["avgManhattanPlanLength"]/res["ManhattanTestSize"])
        mPlotExplored.append(res["avgManhattanExplored"]/res["ManhattanTestSize"])
        mPlotSuccess.append(res["ManhattanTestSize"]/res["testSize"])


    #nnPlotPlan.append(res["avgCNNPlanLength"]/res["CNNTestSize"])
    #nnPlotExplored.append(res["avgCNNExplored"]/res["CNNTestSize"])
    #nnPlotSuccess.append(res["CNNTestSize"]/res["testSize"])


        np.save(str(idx)+"astardictmanhattan.npy", res)    
        print("Medium {}".format(xLevelSize[idx]))
        print("Manhattan Test Size")
        print(res["ManhattanTestSize"])
        print("CNN Test Size")
        print(res["CNNTestSize"])
        print(res["testSize"])
        idx += 1
 


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
