import os
import sys
import subprocess
import argparse
import logging
import glob
import numpy as np
from time import time
import urllib

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.sampler import Sampler

import MinkowskiEngine as ME


class THUMANDataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        self.root = "/data1/tangyingzhi/Project/IntegratedPIFu/pytorch3dthumanpoints/"
        self.persons = []
        for i in range(0,14):
            self.persons.append(str(i).zfill(4))

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):

        mesh_file = self.root + self.persons[idx] + "_10w.ply"
        gsattribute = np.load("/public/sdc/tangyingzhi/gaussian-splatting/voxelgs/3dgs" + self.persons[idx] + "voxel.npz")
        person = self.persons[idx]
        shs = torch.from_numpy(gsattribute['shs'])[:,:,0,:]
        ptsnum = shs.shape[1]
        # print(ptsnum)
        shs = shs.reshape(1,ptsnum,3)
        scale = torch.from_numpy(gsattribute['scale'])
        # print(scale.shape)
        # print(scale.max())
        rotation = torch.from_numpy(gsattribute['rotation'])
        # print(rotation.shape)
        opacity = torch.from_numpy(gsattribute['opacity'])
        occupancy = torch.ones_like(opacity)
        # print(opacity.shape)
        # print(opacity.min())
        # print(opacity.max())
        feature1 = torch.cat([shs, occupancy],2)
        
        pcd = o3d.io.read_point_cloud(mesh_file)
        vertices = np.asarray(pcd.points)
        orivert = vertices

        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        vertices =(vertices - vmin) / (vmax - vmin).max()
        xyz = vertices * 512
        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
        std = np.array(vmax - vmin, dtype='float32').max()
        # print(std)

        mean = np.array(vmin, dtype='float32')
        # print(mean)

        return (coords, xyz[inds], idx, feature1, std, mean, person)
