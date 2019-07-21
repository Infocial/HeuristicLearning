from simulator.sokoban_world import SokobanWorld
import console_utils
import numpy as np
import settings
import argparse
import utils
import time
import sys
import os
from random import Random
# Net class
from var_sokogcnspmv import GCN
# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Deep graph library
import dgl
import networkx as nx


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
PAUSE_TIME = 1.0 # time between time-steps, in seconds


# Auto generated params:
TEMP_DIR = "/tmp"

# Load data and initialize grid world
f = utils.load_dataset(args.dataset_name, TEMP_DIR)
X, Y = f.attrs["world_shape"]
env = SokobanWorld()
#env.render_MPL()


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
    

    # Use GPU or CPU
    device = torch.device("cuda:0")
    # device = torch.device("CPU")

    if args.grouped:
        indices = np.reshape(indices, [f.attrs["num_env_instances"], f.attrs["num_variations_per_instance"]])
        print(indices)
    if args.randomize:
        np.random.shuffle(indices)
    if args.grouped:
        indices = indices.flatten()

    # Split dataset into train and test with different environements
    # Splitting index instead of actual data
    
    # Train percentage split 
    t_p = 0.9
    train_split = t_p*f.attrs["num_samples"]
    test_split = train_split+f.attrs["num_samples"]*(1.0-t_p)/2
    train_index = np.linspace(0,int(train_split),int(train_split),dtype=int)
    #train_index = np.linspace(0,6,6,dtype=int)
    test_index = indices[int(train_split):int(test_split)]
    val_index = indices[int(test_split):]

    # Test for different environements in each data split
    print("Are training dataset with the same env? {}".format(len(train_index)%f.attrs["num_variations_per_instance"]==0))
    print("Are testing dataset with the same env? {}".format(len(test_index)%f.attrs["num_variations_per_instance"]==0))
    print("Are validation dataset with the same env? {}".format(len(val_index)%f.attrs["num_variations_per_instance"]==0))

    print("Datapoints for training: {}".format(2*f["action"][:len(train_index)].size))

    # Shuffle the training data
    # print("Train indices before shuffle {}".format(train_index[:10]))
    # Random(42).shuffle(train_index)
    # print("Train indices after shuffle {}".format(train_index[:10]))
    
    def createGraph(graph, imglen):
        
        g = graph
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

        g.add_edges(src,dst)
        g.add_edges(dst,src)
        

        # show the network
        # import networkx as nx 
        # import matplotlib.pyplot as plt
        # fig = plt.figure(dpi=150)
        # nx_g = g.to_networkx().to_undirected()
        # pos = nx.spectral_layout(nx_g)
        # nx.draw(nx_g, pos, with_labels=True, node_color=[[.7,.7,.7]])
        # plt.show()
        
        return g
    
    def noWallGraph(start_img, goal_img, imglen):
        featurearray = np.array(np.concatenate((start_img[:,:,1:],goal_img[:,:,1:]),axis=2),dtype="Float32")
        featurelist = []
        #numberNodes = imglen**2-len(np.nonzero(start_img[:,:,0]))
        #nodes = range(numberNodes)
        v = {}
        idx = 0
        src = []
        dst = []

        for i in range(1,imglen-1):
            for k in range(1,imglen-1):
                if start_img[i,k,0] == 0:
                    v[(i,k)] = idx
                    featurelist.append(featurearray[i,k,:])
                    idx += 1
                    if start_img[i-1,k,0] == 0:
                        src.append((i,k))
                        dst.append((i-1,k))
                    if start_img[i+1,k,0] == 0:
                        src.append((i,k))
                        dst.append((i+1,k))
                    if start_img[i,k-1,0] == 0:
                        src.append((i,k))
                        dst.append((i,k-1))
                    if start_img[i,k+1,0] == 0:
                        src.append((i,k))
                        dst.append((i,k+1))
        
        edges_src = []
        for tup in src:
            edges_src.append(v[tup])
        edges_dst = []
        for tup in src:
            edges_dst.append(v[tup])
        

        nG = nx.Graph()
        nG.add_nodes_from(range(idx))
    
        g = dgl.DGLGraph(nG)
        g.add_edges(edges_src,edges_dst)
        g.add_edges(edges_dst,edges_src)
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(device)
        g.ndata['norm'] = norm.unsqueeze(1)

        g.ndata['feat'] = torch.from_numpy(np.array(featurelist, dtype="Float32"))

        return g
    
    def getFeatures(start_img, goal_img):
        img = torch.from_numpy(np.array(np.concatenate((start_img,goal_img),axis=2).reshape(X**2,8),dtype="Float32"))
        # featurelist = []
        # for i in range(len(start_img)):
        #     for k in range(len(start_img)):
        #         #graph.nodes[i*k].data['feat'] = img[i,k,:]
    
        return img
    

    # Graph is the same for every map only features are different
    # but not in this version
    #nG = nx.Graph()
    #nG.add_nodes_from(range(X**2))
    #g = createGraph(X)
    # self loops
    
    # g = createGraph(dgl.DGLGraph(nG), X)
    # g.add_edges(g.nodes(), g.nodes())
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # norm = norm.to(device)
    # g.ndata['norm'] = norm.unsqueeze(1)
    # TODO: use function no wall graph to create graph with features in loop
    # Network
    
    inputfeat = 6
    hiddenfeat = 12
    layers = 12
    dropout = 0.0
    net = GCN(inputfeat,hiddenfeat,layers,F.relu, dropout)

    net.to(device)

    # Training params
    learning_rate = 0.00001
    minibatch = 1


    # Optimiser and loss function
    lossfunction = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(net.parameters(),lr=learning_rate, betas=(0.9,0.999),eps=1e-8)
    
    

    # Internal values
    number_of_batches = 0
    mb_idx = 0
    batch_input = []
    batch_target = []

    running_loss = 0.0
    train_loss = 0.0

    # Test values
    testBatch_input = []
    testBatch_target = []
    tb_idx = 0
    test_loss = 0.0
    testLossfunction = nn.L1Loss(reduction="mean")
    # For gpu memory
    lossbatch = 1
    number_of_lossbatches = 0

    # Designed for bootstrapping data
    def drawUniform(upperBound):
        bd = np.random.randint(0,high=upperBound,size=(upperBound,2))
        for i in bd:
            if (i[0] > i[1]):
                temp = i[1]
                i[1] = i[0]
                i[0] = temp
            elif (i[0]==i[1]):
                if (i[1]==upperBound-1):
                    i[0] += -1
                else:
                    i[1] += 1
        return bd


    def batchActions():
        global mb_idx 
        global number_of_batches 
        global batch_input 
        global batch_target
        global net 
        global running_loss
        global optimizer
        global lossfunction
        global train_loss

        # Batch to tensor
        g = batch_input[0]
        ptb_input = g.ndata['feat'].to(device)        
        ptb_target = torch.from_numpy(np.array(batch_target,dtype="float32")).to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # perform network step
        # Forward pass
        y_pred = net(ptb_input, g).view(minibatch)

        # Compute loss
        loss = lossfunction(y_pred, ptb_target)
        running_loss += loss.item()
        train_loss += loss.item()
        
        # Backward pass, update weights.
        loss.backward()
        optimizer.step()

        # if(loss.item() > 1000):
        #     print(loss.item())
        #     print(running_loss)
        #     print(running_loss/2000)
        #     print(y_pred)
        #     print(ptb_target)

        # Reset batch
        mb_idx = 0
        number_of_batches = number_of_batches + 1
        batch_input = []
        batch_target = []

        if (number_of_batches < 100 and number_of_batches % 5 == 1):
            print("Running loss for batch {} is {}".format(number_of_batches,running_loss/number_of_batches))
        
        if (number_of_batches % 2000 == 0):
            print("Running loss for batch {} is {}".format(number_of_batches,running_loss/2000))
 
            running_loss = 0.0
        #print("BATCH ACTION")


    def testAction():

        global testBatch_input
        global testBatch_target
        global net
        global test_loss
        global testLossfunction
        global number_of_lossbatches
        global tb_idx


        # Batch to tensor
        g = testBatch_input[0]
        tb_input = g.ndata['feat'].to(device)        
        
        tb_target = torch.from_numpy(np.array(testBatch_target,dtype="float32")).to(device)


        # perform network step
        # Forward pass
        y_pred = net(tb_input, g).view(lossbatch)
        
        tloss = testLossfunction(y_pred, tb_target)

        test_loss += tloss.item()

        # Reset batch
        tb_idx = 0
        number_of_lossbatches = number_of_lossbatches + 1
        testBatch_input = []
        testBatch_target = []

    net.train()
    for index,i in enumerate(train_index):
        
        #print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
        T =  f["sequence_length"][i]

        # For bootstrapping
        bootstrap = []
        indices_bootstrap = drawUniform(T)

        
        env.reset_world(f["wall"][i], f["robot_loc"][i,0], f["obj_loc"][i,:,0], f["goal_loc"][i])

        # [:,:,0] is wall layer, 
        # [:,:,1] is goal layer,
        # [:,:,2] is object layer, 
        # [:,:,3] is robot layer
        init_img_state = env.get_observation()
        goal_img_state = init_img_state.copy()
        goal_img_state[:,:,3] *= False # Robot layer is zeroed out
        goal_img_state[:,:,2] = goal_img_state[:,:,1] # Objects should be at goal

        # add to bootstrap
        bootstrap.append(init_img_state)

        # Change to graph and add to batch
        
        batch_input.append(noWallGraph(init_img_state,goal_img_state,X))
        batch_target.append(T)
        mb_idx += 1


        if mb_idx == minibatch:
            batchActions()


        idx0, idx1 = range(T), f["action"][i,:T]
        #print(f["action"].size)
        #print("idx0 {}, idx1 {}".format(idx0,idx1))
        action_one_hot = np.zeros([T, num_actions], dtype='bool')
        action_one_hot[idx0, idx1] = 1
        #print("action one hot {}".format(action_one_hot))

        
        for t in range(T):
            
            env.step(action_one_hot[t])
            step_img_state = env.get_observation()
            bootstrap.append(step_img_state)
            batch_input.append(noWallGraph(step_img_state,goal_img_state,X))
            batch_target.append(T-1-t)
            mb_idx += 1

            if mb_idx == minibatch:
                batchActions()

        
        # Bootstrap loop
        
        # for k in indices_bootstrap:
        #     goal_step = bootstrap[k[1]].copy()
        #     #goal_step[:,:,3] *= False
        #     batch_input.append(np.concatenate((bootstrap[k[0]],goal_step),axis=2))
        #     batch_target.append(k[1]-k[0]+1)
        #     mb_idx += 1

        #     if mb_idx == minibatch:
        #         batchActions()
    
    # Calculate the loss
    print("Done with all the training")
    print("Trained on {} datapoints".format(number_of_batches*minibatch))

    # Calculate test loss
    net.eval()
    for index,i in enumerate(test_index):
        
        if (i == 42000):
            print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
        T =  f["sequence_length"][i]

        # For bootstrapping
        # bootstrap = []
        # indices_bootstrap = drawUniform(T)

        
        env.reset_world(f["wall"][i], f["robot_loc"][i,0], f["obj_loc"][i,:,0], f["goal_loc"][i])

        # [:,:,0] is wall layer, 
        # [:,:,1] is goal layer,
        # [:,:,2] is object layer, 
        # [:,:,3] is robot layer
        init_img_state = env.get_observation()
        goal_img_state = init_img_state.copy()
        goal_img_state[:,:,3] *= False # Robot layer is zeroed out
        goal_img_state[:,:,2] = goal_img_state[:,:,1] # Objects should be at goal

        # add to bootstrap
        # bootstrap.append(init_img_state)

        # add to batch
        #print(init_img_state.shape)
        #np.concatenate((np.array(init_img_state),np.array(goal_img_state)))
        testBatch_input.append(noWallGraph(init_img_state,goal_img_state,X))
        testBatch_target.append(T)
        tb_idx += 1


        if tb_idx == lossbatch:
            testAction()
        

        idx0, idx1 = range(T), f["action"][i,:T]
        #print(f["action"].size)
        #print("idx0 {}, idx1 {}".format(idx0,idx1))
        action_one_hot = np.zeros([T, num_actions], dtype='bool')
        action_one_hot[idx0, idx1] = 1
        #print("action one hot {}".format(action_one_hot))

        
        for t in range(T):
            
            env.step(action_one_hot[t])
            step_img_state = env.get_observation()
            # bootstrap.append(step_img_state)
            testBatch_input.append(noWallGraph(step_img_state,goal_img_state,X))
            testBatch_target.append(T-1-t)
            tb_idx += 1

            if tb_idx == lossbatch:
                testAction()
        
        # Bootstrap loop
        
        # for k in indices_bootstrap:
        #     goal_step = bootstrap[k[1]].copy()
        #     #goal_step[:,:,3] *= False
        #     batch_input.append(np.concatenate((bootstrap[k[0]],goal_step)))
        #     batch_target.append(k[1]-k[0]+1)
        #     mb_idx += 1

        #     if mb_idx == minibatch:
        #         batchActions()
    print("Network features gcn")
    print("hidden features {}".format(hiddenfeat))
    print("layers {}".format(layers))
    print("learning rate {}".format(learning_rate))
    print("dropout {}".format(dropout))
    print(train_loss/number_of_batches)
    print(test_loss/number_of_lossbatches)
    print("done")

       
finally:
    filepath = f.filename
    f.close()
    if filepath.startswith(TEMP_DIR):
        os.remove(filepath)
