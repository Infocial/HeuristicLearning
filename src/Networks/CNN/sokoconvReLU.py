import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class SokoNet(nn.Module):
    def __init__(self):
        super(SokoNet, self).__init__()
        self.conv1 = nn.Conv2d(8,64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv8 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv9 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv13 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        self.conv14 = nn.Conv2d(72,64,kernel_size=3,stride=1,padding=1)
        
        # Dimension in without window 64*11*11
        # Dimension with window is 64*window_width*window_height 
        self.fc1 = nn.Linear(64,256)
        self.fc2 = nn.Linear(256,1)
    
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        
        y = F.relu(self.conv2(self.skipConnections(x,y)))
        y = F.relu(self.conv3(self.skipConnections(x,y)))
        y = F.relu(self.conv4(self.skipConnections(x,y)))
        y = F.relu(self.conv5(self.skipConnections(x,y)))
        y = F.relu(self.conv6(self.skipConnections(x,y)))
        y = F.relu(self.conv7(self.skipConnections(x,y)))
        y = F.relu(self.conv8(self.skipConnections(x,y)))
        y = F.relu(self.conv9(self.skipConnections(x,y)))
        y = F.relu(self.conv10(self.skipConnections(x,y)))
        y = F.relu(self.conv11(self.skipConnections(x,y)))
        y = F.relu(self.conv12(self.skipConnections(x,y)))
        y = F.relu(self.conv13(self.skipConnections(x,y)))
        y = F.relu(self.conv14(self.skipConnections(x,y)))
        
        y = self.robotPosWindow(x,y)
        y = y.view(y.size(0),-1)
        #print("This is the output size {}".format(y.size()))
        y = F.relu(self.fc1(y))
        return self.fc2(y)


    def skipConnections(self,inputData,prevLayer):
        return torch.cat((inputData,prevLayer),1)

    def robotPosWindow(self,inputData,layerData):
        robotpositions = torch.nonzero(inputData[:,3,:,:])
        temp = []
        
        for pos in robotpositions:
            temp.append(layerData[pos[0], :, pos[1], pos[2]])
        
        return torch.stack(temp)
       
