#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def load_data(partition):
    DATA_DIR = 'sparse.npy'
    
    # Load data from the .npy file
    data = np.load(DATA_DIR, allow_pickle=True)
    
    # Print the loaded data to understand its structure
    print("Loaded data:", data)
    
    # Check if the data is a structured array or a regular array
    if isinstance(data, np.ndarray):
        # If it's a regular array, you might need to adjust how you access the data
        all_data = data[:, :-1].astype('float32')  # Assuming last column is label
        all_label = data[:, -1].astype('int64')     # Assuming last column is label
    else:
        # If it's a structured array, access it using the keys
        all_data = data['data'].astype('float32')  # Adjust based on actual structure
        all_label = data['label'].astype('int64')  # Adjust based on actual structure

    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[128])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[128])
    
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)