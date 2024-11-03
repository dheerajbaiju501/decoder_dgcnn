
import pdb

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    
    print(x.shape)
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.contiguous().reshape(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    else:
        idx = knn(x[:, 6:], k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    print("shape of x = ",x.shape)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(2)
        self.bn5 = nn.BatchNorm1d(2)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(500)
        self.bn2 = nn.BatchNorm2d(684)
        self.bn3 = nn.BatchNorm2d(300)
        self.bn4 = nn.BatchNorm2d(500)
        self.bn5 = nn.BatchNorm1d(400)

        self.conv1 = nn.Sequential(nn.Conv2d(500, 500, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(684, 684, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(300, 300, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(500, 500, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn5 = nn.BatchNorm1d(400)
                           
        self.conv5 = nn.Sequential(nn.Conv1d(400, 400, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(2 * 400, 400, bias=False)
        self.bn6 = nn.BatchNorm1d(400)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        print("Original tensor size:", x.size())
        #x = x.reshape(batch_size, 1000)  # (batch_size, 3, 2048)
        
        x = x.repeat(1, 171, 1, 1) 
        
       
        
        x = get_graph_feature(x, k=self.k)
        x = x[:,:500,:,:]
        #pdb.set_trace()
        print(x.shape)
        
        x = self.conv1(x)  # (batch_size, 64, 2048, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, 2048)

        x = get_graph_feature(x1, k=self.k)
        
        print("Shape of input tensor before line 139:", x.shape)  # Debugging line
        if x.shape[1] == 8192:  # Check if the input has 8192 channels
            x = x[:, :128, :, :]
        
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        print("shape of x2 = ",x2.shape)
        x = get_graph_feature(x2, k=self.k)
        x = x[:,:300,:,:]
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        #x = x.repeat(1, 2, 1, 1)
        x = x[:,:500,:,:]
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        print("Shape of tensor 1:", x1.shape)  # Replace tensor1 with your actual tensor variable names
        print("Shape of tensor 2:", x2.shape)  # Replace tensor2 with your actual tensor variable names
        print("Shape of tensor 1:", x3.shape)  # Replace tensor1 with your actual tensor variable names
        print("Shape of tensor 2:", x4.shape)  # Replace tensor2 with your actual tensor variable names
        x1_adj = x1[:,:64,:64]
        x2 = x2[:,:,:64]
        print("shape of x1_adj = ",x1_adj.shape)
        print("shape of x2 = ",x2.shape)
        result1 = torch.cat((x1_adj, x2), dim=1)
        x4_adjusted = x4[:, :128, :64]
        x3 = x3[:,:,:64]
        print("shape of x3 = ",x3)
        print("shape of x4_adj = ",x4_adjusted.shape)
        x = torch.cat((x3, x4_adjusted), dim=1)  
        
        print("Shape of input tensor before line 164:", x.shape)  # Debugging line
        #x = x.repeat(1, 2, 1)
        if x.shape[1] == 256:  # Check if the input has 256 channels
            x = x.repeat(1, 2, 1)
        x = x[:,:400,:]
        x = self.conv5(x)  # (batch_size, 2, 2048)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 2)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 2)
        x = torch.cat((x1, x2), 1)  # (batch_size, 2048)

        #x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        #x = self.dp1(x)
        #x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        #x = self.dp2(x)
        #x = self.linear3(x)
    
        return x

        

class Decoder(nn.Module):
    def __init__(self, latent_dim, grid_size):
        super(Decoder, self).__init__()
        self.grid_size = grid_size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(latent_dim + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1)
        )
        
       

    def forward(self,latent_vector):
        grid = torch.meshgrid(torch.linspace(-1, 1, self.grid_size), torch.linspace(-1, 1, self.grid_size))
        grid = torch.stack(grid, dim=-1).view(-1, 2)  # (N_grid * N_grid, 2)
        grid = grid.unsqueeze(0).repeat(latent_vector.size(0), 1, 1).to(latent_vector.device)
        latent_vector = latent_vector.unsqueeze(2).repeat(1, 1, grid.size(1))
        input = torch.cat((grid.transpose(2, 1), latent_vector), dim=1)
        input = torch.nn.functional.pad(input, (0, 0, 0, 1026 - 802, 0, 0))
        print("shape of input = ",input.shape)
        reconstructed_points = self.mlp1(input)
        return reconstructed_points

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.encoder = DGCNN(args)
        self.decoder = Decoder(latent_dim=args.emb_dims, grid_size=45)

    def forward(self, x):
        latent_vector = self.encoder(x)  
        reconstructed_points = self.decoder(latent_vector)  
        return reconstructed_points

