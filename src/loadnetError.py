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
from sokoconvReLU import SokoNet
#from chebnetskip import GCN
# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Charts
import matplotlib
import matplotlib.pyplot as plt


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
    dim = f.attrs["world_shape"][0]

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
    test_index = indices[int(train_split):int(test_split)]
    val_index = indices[int(test_split):]

    # Test for different environements in each data split
    print("Are training dataset with the same env? {}".format(len(train_index)%f.attrs["num_variations_per_instance"]==0))
    print("Are testing dataset with the same env? {}".format(len(test_index)%f.attrs["num_variations_per_instance"]==0))
    print("Are validation dataset with the same env? {}".format(len(val_index)%f.attrs["num_variations_per_instance"]==0))

    print("Datapoints for training: {}".format(2*f["action"][:len(train_index)].size))

    def createGraph(imglen):
        
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

    def getFeatures(start_img, goal_img):
        img = torch.from_numpy(np.array(np.concatenate((start_img,goal_img),axis=2).reshape(X**2,8),dtype="Float32"))
        # featurelist = []
        # for i in range(len(start_img)):
        #     for k in range(len(start_img)):
        #         #graph.nodes[i*k].data['feat'] = img[i,k,:]
    
        return img
    # Shuffle the training data
    # print("Train indices before shuffle {}".format(train_index[:10]))
    # Random(42).shuffle(train_index)
    # print("Train indices after shuffle {}".format(train_index[:10]))
    
    # Network
    net = SokoNet()
    g = createGraph(X).to(device)
    #net = GCN()
    net.load_state_dict(torch.load("cnnrelu.pth"))
    #net.load_state_dict(torch.load("gcncheb4skipcontest.pth"))
    net.to(device)
    net.eval()
    filenameout = "gcncheb4skip-med9"
    # Training params
    #learning_rate = 0.00009
    #minibatch = 128

    # Optimiser and loss function
    #lossfunction = nn.MSELoss(reduction="mean")
    #optimizer = optim.Adam(net.parameters(),lr=learning_rate, betas=(0.95,0.99999),eps=1e-8)
    
    testlength = len(test_index)
    ErrPerSolLength = {}
    Targets = {}

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
        bd = np.random.randint(0,high=upperBound,size=(3*upperBound,2))
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
        ptb_input = torch.from_numpy(np.array(batch_input,dtype="float32")).to(device)
        ptb_input = ptb_input.permute(0,3,1,2)
        ptb_target = torch.from_numpy(np.array(batch_target,dtype="float32")).to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # perform network step
        # Forward pass
        y_pred = net(ptb_input).view(minibatch)

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
        #tb_input = torch.from_numpy(np.array(testBatch_input,dtype="float32")).to(device)
        #tb_input = tb_input.permute(0,3,1,2)
        tb_input = testBatch_input[0].to(device)
        tb_target = torch.from_numpy(np.array(testBatch_target,dtype="float32")).to(device)


        # perform network step
        # Forward pass
        y_pred = net(tb_input).view(lossbatch)
        
        tloss = testLossfunction(y_pred, tb_target)

        test_loss += tloss.item()

        for i in range(lossbatch):
            #rooy_pred = round(y_pred[i].item())
            if tb_target[i].item() in ErrPerSolLength:
                ErrPerSolLength[tb_target[i].item()].append(tloss.item())
                Targets[tb_target[i].item()] += 1
            else:
                ErrPerSolLength[tb_target[i].item()] = [tloss.item()]
                Targets[tb_target[i].item()] = 1
                

        # Reset batch
        tb_idx = 0
        number_of_lossbatches = number_of_lossbatches + 1
        testBatch_input = []
        testBatch_target = []

    sumTargets = 0
    nTargets = 0
    # for index,i in enumerate(test_index[:40]):
        
    #     if (i == 42000):
    #         print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
    #     T =  f["sequence_length"][i]
    #     if T < 35:
    #         sumTargets += T
    #         nTargets += 1
    #         if T in Targets:
    #             Targets[T] += 1
    #         else:
    #             Targets[T] = 1

    for index,i in enumerate(test_index[:100]):
        
        if (i == 42000):
            print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
        T =  f["sequence_length"][i]
        if T < 35:
            sumTargets += T
            nTargets += 1
            if T in Targets:
                Targets[T] += 1
            else:
                Targets[T] = 1

    print(sumTargets/nTargets)
    # fig, ax1 = plt.subplots()
    # print(len(indices))
    # #ax1.bar(predictions.keys(), predictions.values(), width=0.8)
    # # ax1.set_title('Predictions')
    # ax1.bar(Targets.keys(), Targets.values(), width=0.6)
    # ax1.set_title('Solution Distribution For Size 18 Manhattan')
    # ax1.set_xlabel('Solution length')
    # ax1.set_ylabel('Number of Samples')
    # plt.tight_layout()
    # plt.show()
    # Calculate test loss
    for index,i in enumerate(test_index):
        
        if (i == 42000):
            print ("Executing {}/{}".format(i,f.attrs["num_samples"]))
        T =  f["sequence_length"][i]
        
        Targets[T]
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
        testBatch_input.append(getFeatures(init_img_state,goal_img_state))
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
            testBatch_input.append(getFeatures(step_img_state,goal_img_state))
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
    #print(train_loss/number_of_batches)
    
    np.save(filenameout+".npy", ErrPerSolLength)
    for key in ErrPerSolLength:
        ErrPerSolLength[key] = np.mean(ErrPerSolLength[key])
    lists = sorted(ErrPerSolLength.items())
    x, y = zip(*lists)
    plt.figure()
    #fig, (ax1, ax2) = plt.subplots(2)
    plt.plot(x, y)
    plt.title('Error per solution length')
    plt.xlabel('Solution length')
    plt.ylabel('L1 loss')
    #ax1.bar(ErrPerSolLength.keys(), ErrPerSolLength.values(), width=0.8)
    #ax1.set_title('Predictions')
    #ax2.bar(targets.keys(), targets.values(), width=0.8)
    #ax2.set_title('Targets')
    plt.tight_layout()
    plt.savefig(filenameout+".png")
    plt.savefig(filenameout+".svg")
    print(test_loss/number_of_lossbatches)
    print("done")
    
finally:
    filepath = f.filename
    f.close()
    if filepath.startswith(TEMP_DIR):
        os.remove(filepath)
