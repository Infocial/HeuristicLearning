# Plots Astar results produced by using dictionaries from other Astar scripts
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop

reslistCNN = []
reslistCHEB = []
reslistMan = []

filenameout = "plotAstar"
dirParts = "partsManhattan/"
fnameparts9 = ["9partsastardictmanhattan.npy"]
fnameparts12 = ["12partsastardictmanhattan20.npy", "12partsastardictmanhattan40.npy", "12partsastardictmanhattan60.npy", "12partsastardictmanhattan80.npy", "12partsastardictmanhattan100.npy"]
fnameparts15 = ["15partsastardictmanhattan20.npy", "15partsastardictmanhattan40.npy", "15partsastardictmanhattan60.npy", "15partsastardictmanhattan80.npy", "15partsastardictmanhattan100.npy"]
fnameparts18 = ["18partsastardictmanhattan10.npy","18partsastardictmanhattan20.npy", "18partsastardictmanhattan30.npy", "18partsastardictmanhattan40.npy", "18partsastardictmanhattan60.npy", "18partsastardictmanhattan80.npy", "18partsastardictmanhattan90.npy", "18partsastardictmanhattan100.npy"]

reslistMan.append(np.load(dirParts+fnameparts9[0], allow_pickle=True).item())

fname12dict = {}
fname12dict["avgManhattanPlanLength"] = 0
fname12dict["avgManhattanExplored"] = 0
fname12dict["ManhattanTestSize"] = 0
fname12dict["testSize"] = 0
fname12dict["avgPlanLength"] = 0
fname12dict["avgCNNPlanLength"] = 0
fname12dict["CNNTestSize"] = 0
fname12dict["avgCNNExplored"] = 0
for fname in fnameparts12:
    temp = np.load(dirParts+fname, allow_pickle=True).item()
    for key in temp:
        fname12dict[key] += temp[key]
reslistMan.append(fname12dict)

fname15dict = {}
fname15dict["avgManhattanPlanLength"] = 0
fname15dict["avgManhattanExplored"] = 0
fname15dict["ManhattanTestSize"] = 0
fname15dict["testSize"] = 0
fname15dict["avgPlanLength"] = 0
fname15dict["avgCNNPlanLength"] = 0
fname15dict["CNNTestSize"] = 0
fname15dict["avgCNNExplored"] = 0
for fname in fnameparts15:
    temp = np.load(dirParts+fname, allow_pickle=True).item()
    for key in temp:
        fname15dict[key] += temp[key]
reslistMan.append(fname15dict)

fname18dict = {}
fname18dict["avgManhattanPlanLength"] = 0
fname18dict["avgManhattanExplored"] = 0
fname18dict["ManhattanTestSize"] = 0
fname18dict["testSize"] = 0
fname18dict["avgPlanLength"] = 0
fname18dict["avgCNNPlanLength"] = 0
fname18dict["CNNTestSize"] = 0
fname18dict["avgCNNExplored"] = 0
for fname in fnameparts18:
    temp = np.load(dirParts+fname, allow_pickle=True).item()
    for key in temp:
        fname18dict[key] += temp[key]
reslistMan.append(fname18dict)


for i in range(4):
    reslistCNN.append(np.load(str(i)+"astardict.npy", allow_pickle=True).item())

for i in range(4):
    reslistCHEB.append(np.load(str(i)+"astardictcheb.npy", allow_pickle=True).item())



#fig, (ax1, ax2) = plt.subplots(2)
mPlotPlan = []
nnPlotPlan = []
chebPlotPlan = []
mPlotExplored = []
nnPlotExplored = []
chebPlotExplored = []
mPlotSuccess = []
nnPlotSuccess = []
chebPlotSuccess = []

xLevelSize = [9,12,15,18]
idx = 0

for res in reslistCNN:
    #mPlotPlan.append(res["avgManhattanPlanLength"]/res["ManhattanTestSize"])
    #mPlotExplored.append(res["avgManhattanExplored"]/res["ManhattanTestSize"])
    #mPlotSuccess.append(res["ManhattanTestSize"]/res["testSize"])

    nnPlotPlan.append(res["avgCNNPlanLength"]/res["CNNTestSize"])
    nnPlotExplored.append(res["avgCNNExplored"]/res["CNNTestSize"])
    nnPlotSuccess.append(res["CNNTestSize"]/res["testSize"])
    
    print("Medium {}".format(xLevelSize[idx]))
    #print("Manhattan Test Size")
    #print(res["ManhattanTestSize"])
    print("CNN Test Size")
    print(res["CNNTestSize"])
    print(res["testSize"])
    idx += 1

idx = 0
for res in reslistCHEB:
    #mPlotPlan.append(res["avgManhattanPlanLength"]/res["ManhattanTestSize"])
    #mPlotExplored.append(res["avgManhattanExplored"]/res["ManhattanTestSize"])
    #mPlotSuccess.append(res["ManhattanTestSize"]/res["testSize"])

    chebPlotPlan.append(res["avgCNNPlanLength"]/res["CNNTestSize"])
    chebPlotExplored.append(res["avgCNNExplored"]/res["CNNTestSize"])
    chebPlotSuccess.append(res["CNNTestSize"]/res["testSize"])
    
    print("Medium {}".format(xLevelSize[idx]))
    #print("Manhattan Test Size")
    #print(res["ManhattanTestSize"])
    print("GCN Test Size")
    print(res["testSize"])

    idx += 1

idx = 0
for res in reslistMan:
    mPlotPlan.append(res["avgManhattanPlanLength"]/res["ManhattanTestSize"])
    mPlotExplored.append(res["avgManhattanExplored"]/res["ManhattanTestSize"])
    mPlotSuccess.append(res["ManhattanTestSize"]/res["testSize"])

    #chebPlotPlan.append(res["avgCNNPlanLength"]/res["CNNTestSize"])
    #chebPlotExplored.append(res["avgCNNExplored"]/res["CNNTestSize"])
    #chebPlotSuccess.append(res["CNNTestSize"]/res["testSize"])
    
    print("Medium {}".format(xLevelSize[idx]))
    print("Manhattan Test Size")
    print(res["ManhattanTestSize"])
    #print("GCN Test Size")
    #print(res["testSize"])
    idx += 1
 

print(nnPlotPlan)
print(chebPlotPlan)
print(mPlotPlan)
ind = np.arange(len(xLevelSize))
xt = ["9","12","15","18"]
# Plot everything
# Plot solution length
fig1, ax1 = plt.subplots()

ax1.plot(xLevelSize, mPlotPlan, label="A* M")
ax1.plot(xLevelSize, nnPlotPlan, label="A* CNN")
ax1.plot(xLevelSize, chebPlotPlan, label="A* GCN")

ax1.set_title('Solution Length Per Level Size')
ax1.set_ylabel('Solution Length')
ax1.set_xlabel('Level Size')
ax1.set_xticks(xLevelSize)
ax1.set_xticklabels(xt)

ax1.legend()

fig1.tight_layout()
fig1.savefig(filenameout+"Plan.png")
fig1.savefig(filenameout+"Plan.svg")


# Plot nodes explored
fig2, ax2 = plt.subplots()

ax2.plot(xLevelSize, mPlotExplored, label="A* M")
ax2.plot(xLevelSize, nnPlotExplored, label="A* CNN")
ax2.plot(xLevelSize, chebPlotExplored, label="A* GCN")

ax2.set_title('Nodes Explored Per Level Size')
ax2.set_ylabel('Avg. Explored Nodes')
ax2.set_yscale('log')
ax2.set_xlabel('Level Size')
ax2.set_xticks(xLevelSize)
ax2.set_xticklabels([str(x) for x in xLevelSize])

ax2.legend()

fig2.tight_layout()
fig2.savefig(filenameout+"Explored.png")
fig2.savefig(filenameout+"Explored.svg")


# Plot success rate

fig, ax = plt.subplots()
barwidth = 0.35

ind = np.arange(len(xLevelSize))

ax.bar(ind-barwidth, mPlotSuccess, barwidth, label="A* M")
ax.bar(ind, nnPlotSuccess, barwidth, label="A* NN")
ax.bar(ind+barwidth, chebPlotSuccess, barwidth, label="A* GCN")

ax.set_title('Success Rate Per Level Size')
ax.set_ylabel('Success Rate')
ax.set_xlabel('Level Size')
ax.set_xticks(ind)
ax.set_xticklabels([str(x) for x in xLevelSize])
ax.legend()

fig.tight_layout()
fig.savefig(filenameout+"Success.png")
fig.savefig(filenameout+"Success.svg")
