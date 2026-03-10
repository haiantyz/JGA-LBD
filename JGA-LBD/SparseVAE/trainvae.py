# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
import cv2
from random import randint 
import json
from utils.general_utils import PILtoTorch2
from PIL import Image
from gaussian_renderer.__init__ import render
from argparse import ArgumentParser, Namespace
from plyfile import PlyData, PlyElement
from utils_sparse import isin, sort_sparse_tensor

import subprocess
import argparse
import logging
import glob
import numpy as np
from time import time
import urllib
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
import MinkowskiEngine as ME
from torch.utils.data.sampler import Sampler
from arguments import ModelParams, PipelineParams, OptimizationParams
# from examples.reconstruction import InfSampler, resample_mesh
from thumandatasetshs import THUMANDataset
from autoencoder import InceptionResNet, make_layer
import lpips

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the fixed ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################
def PointCloud(points, colors=None):
    points= points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("testvae.ply", pcd)
    return pcd


def collate_pointcloud_fn(list_data):
    coords, feats, std, mean, person = list(zip(*list_data))
    # print(std[0])
    # Concatenate all lists
    # person = person_name[:4]
    # print(person)
    # angle = person_name[5:]
    # return {
    #     "coords": ME.utils.batched_coordinates(coords),
    #     "xyzs": [torch.from_numpy(feat).float() for feat in feats],
    #     "labels": torch.LongTensor(labels),
    #     'shsfeature': torch.from_numpy(np.array([sfeat.numpy() for sfeat in shsfeatures])),
    #     'std': torch.from_numpy(np.array([std0 for std0 in std])),
    #     'mean': torch.from_numpy(np.array([mean0 for mean0 in mean])),
    #     'person': torch.from_numpy(np.array([person0 for person0 in person])),
    # }
    
    asda = torch.Tensor([])
    for i in feats:
        asda = torch.cat([asda, i],0)
        # print(i.shape)
    # asda = asda.squeeze()
    # asda = np.array(asda)
    # asda = torch.from_numpy(asda)
    return {
        "coords": ME.utils.batched_coordinates(coords),
        'person': person,
        'shsfeature': asda,
        'std': std,
        'mean': mean
    }



def make_data_loader(
    phase, augment_data, batch_size, shuffle, num_workers, repeat, config
):
    # dset = ModelNet40Dataset(phase, config=config)
    dset = THUMANDataset(phase, config=config)
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split(".")[0] + " %(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=4000000)
parser.add_argument("--val_freq", type=int, default=500)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="/data/tangyingzhi/data/shsvaeresnet3.pth")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)

###############################################################################
# End of utility functions
###############################################################################


class Encoder(torch.nn.Module):
    def __init__(self, channels=[20,64,256,256,64,8]):
        super().__init__()
        bn_momentum=0.1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=1,
            bias=True,
            dimension=3)
        self.bnconv0 = ME.MinkowskiBatchNorm(channels[1], momentum=bn_momentum)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.bndown0 = ME.MinkowskiBatchNorm(channels[2], momentum=bn_momentum)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=2, 
            channels=channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=2,
            stride=1,
            bias=True,
            dimension=3)
        self.bnconv1 = ME.MinkowskiBatchNorm(channels[2], momentum=bn_momentum)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.bndown1 = ME.MinkowskiBatchNorm(channels[3], momentum=bn_momentum)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=2, 
            channels=channels[3])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=2,
            stride=1,
            bias=True,
            dimension=3)
        self.bnconv2 = ME.MinkowskiBatchNorm(channels[3], momentum=bn_momentum)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.bndown2 = ME.MinkowskiBatchNorm(channels[4], momentum=bn_momentum)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[4])

        self.convmean = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=2,
            stride=1,
            bias=True,
            dimension=3)
        self.bnmean = ME.MinkowskiBatchNorm(channels[5], momentum=bn_momentum)
        self.convvar = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=2,
            stride=1,
            bias=True,
            dimension=3)
        self.bnvar = ME.MinkowskiBatchNorm(channels[5], momentum=bn_momentum)
        self.elu = ME.MinkowskiELU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        out0 = self.elu(self.bndown0(self.down0(self.elu(self.bnconv0(self.conv0(x))))))
        out0 = self.block0(out0)
        out1 = self.elu(self.bndown1(self.down1(self.elu(self.bnconv1(self.conv1(out0))))))
        out1 = self.block1(out1)
        out2 = self.elu(self.bndown2(self.down2(self.elu(self.bnconv2(self.conv2(out1))))))
        out2 = self.block2(out2)

        mean = self.bnmean(self.convmean(out2))
        var = self.bnvar(self.convvar(out2))

        return  mean, var


class DecoderSHS(nn.Module):

    CHANNELS = [8, 64, 128, 256, 128, 64, 64]
    resolution = 128

    def __init__(self):
        nn.Module.__init__(self)
        self.training=True
        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[1], kernel_size=2, stride=1, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[1], ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[2], ch[3], kernel_size=2, stride=1, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[3], ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[4], ch[5], kernel_size=2, stride=1, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block5_cls = ME.MinkowskiConvolution(
            ch[5], 1, kernel_size=1, bias=True, dimension=3
        )
        
        # Block 6
        self.block6 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )
        self.block6color = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], 3, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(3),
            ME.MinkowskiSigmoid(),
        )
        self.block6shs2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], 9, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(9),
            ME.MinkowskiTanh(),
        )
        self.block6scale = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], 3, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(3),
            ME.MinkowskiSigmoid(),
        )
        self.block6rotation = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], 4, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(4),
            ME.MinkowskiSigmoid(),
        )
        self.block6opacity = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], 1, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(1),
            ME.MinkowskiSigmoid(),
        )

        self.block6_cls = ME.MinkowskiConvolution(
            ch[6], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []

        # z = ME.SparseTensor(
        #     features=z_glob.F,
        #     coordinates=z_glob.C,
        #     tensor_stride=self.resolution,
        #     coordinate_manager=z_glob.coordinate_manager,
        # )

        # Block1
        out1 = self.block1(z_glob)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        out1 = self.pruning(out1, keep1)

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()

        if self.training:
            keep2 += target

        # Remove voxels 16
        out2 = self.pruning(out2, keep2)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()

        if self.training:
            keep3 += target

        # Remove voxels 8
        out3 = self.pruning(out3, keep3)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()

        if self.training:
            keep4 += target

        # Remove voxels 4
        out4 = self.pruning(out4, keep4)

        # Block 5
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()

        if self.training:
            keep5 += target
        out5 = self.pruning(out5, keep5)

        

        # Block 5
        out6 = self.block6(out5)
        out6shs = self.block6color(out5)
        out6shs2 = self.block6shs2(out5)
        out6scale = self.block6scale(out5)
        out6rotation = self.block6rotation(out5)
        out6opacity = self.block6opacity(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()
        # print(out6.shape)
        # print(out6shs.shape)
        # Last layer does not require keep
        # if self.training:
        #   keep6 += target

        # Remove voxels 1
        if keep6.sum() > 0:
            out6 = self.pruning(out6, keep6)
            out6shs = self.pruning(out6shs, keep6)
            out6shs2 = self.pruning(out6shs2, keep6)
            out6rotation = self.pruning(out6rotation, keep6)
            out6scale = self.pruning(out6scale, keep6)
            out6opacity = self.pruning(out6opacity, keep6)
            # print(out6.shape)
            # print(out6shs2.shape)

        return out_cls, targets, out6, out6shs, out6shs2, out6rotation, out6scale,  out6opacity
  
    

class AE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = DecoderSHS()

    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar.F)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, sinput, gt_target):
        # zs = self.encoder(sinput)
        means, var = self.encoder(sinput)
        zs = self.reparameterize(means, var)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + var.F - means.F.pow(2) - var.F.exp(),1))
        
        # print(zs.shape)
        # if self.training:
        #     zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout, out6shs, out6shs2, out6rotation, out6scale, out6opacity = self.decoder(zs, gt_target)
        return out_cls, targets, sout, out6shs, out6shs2, out6rotation, out6scale, out6opacity, kl_loss, zs
    


def train(net, dataloader, device, config, pipe):
    # optimizer = optim.SGD(
    #     net.parameters(),
    #     lr=config.lr,
    #     momentum=config.momentum,
    #     weight_decay=config.weight_decay,
    # )
    optimizer = optim.AdamW(
        net.parameters(),
        lr=0.00005
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    MSEloss = nn.MSELoss()
    crit = nn.BCEWithLogitsLoss()

    start_iter = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        print("Resuming weights")
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["curr_iter"]

    net.train()
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_iter = iter(dataloader)
    # val_iter = iter(val_dataloader)
    loss_fnperc = lpips.LPIPS(net='vgg').cuda()

    logging.info(f"LR: {scheduler.get_lr()}")

    for i in range(start_iter, config.max_iter):

        s = time()
        # data_dict = train_iter.next()
        data_dict = next(train_iter)
        d = time() - s

        # optimizer.zero_grad()
        # print(data_dict["shsfeature"].shape)
        # bs = data_dict["shsfeature"].shape[0]
        # ptsnum = data_dict["shsfeature"].shape[2]
        # shsfeaafa = data_dict["shsfeature"].reshape(bs*ptsnum,4)
        sin = ME.SparseTensor(
            features=data_dict["shsfeature"],
            coordinates=data_dict["coords"].int(),
            device=device,
        )
        # print(sin.dense()[0].shape)
        target_key = sin.coordinate_map_key

        
        out_cls, targets, sout, Aout6shs, Aout6shs2, Aout6rotation, Aout6scale, Aout6opacity, kl_loss, zs = net(sin, target_key)
        # print(data_dict['person'])
        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        _, out6shs = Aout6shs.decomposed_coordinates_and_features
        _, out6shs2 = Aout6shs2.decomposed_coordinates_and_features
        _, out6rotation = Aout6rotation.decomposed_coordinates_and_features
        _, out6scale = Aout6scale.decomposed_coordinates_and_features
        _, out6opacity = Aout6opacity.decomposed_coordinates_and_features
        pruning = ME.MinkowskiPruning().cuda()
        maskA = isin(Aout6shs.C, sin.C).to(sout.device)
     
        maskA2 = isin(Aout6shs2.C, sin.C).to(sout.device)
        maskA3 = isin(Aout6scale.C, sin.C).to(sout.device)
        maskA4 = isin(Aout6rotation.C, sin.C).to(sout.device)
        maskA5 = isin(Aout6opacity.C, sin.C).to(sout.device)

        maskB = isin(sin.C, Aout6shs.C).to(sout.device)

        out_intersectshs = pruning(Aout6shs, maskA)
        out_intersectshs2 = pruning(Aout6shs2, maskA)
        out_intersectscale = pruning(Aout6scale, maskA)
        out_intersectrotate = pruning(Aout6rotation, maskA)
        out_intersectopacity = pruning(Aout6opacity, maskA)
        
        gt_intersect = pruning(sin, maskB)
        out_intersectshs = sort_sparse_tensor(out_intersectshs)
        out_intersectshs2 = sort_sparse_tensor(out_intersectshs2)
        out_intersectscale = sort_sparse_tensor(out_intersectscale)
        out_intersectrotate = sort_sparse_tensor(out_intersectrotate)
        out_intersectopacity = sort_sparse_tensor(out_intersectopacity)
        
        gt_intersect = sort_sparse_tensor(gt_intersect)
        assert(gt_intersect.C==out_intersectshs.C).all()
        assert(gt_intersect.C==out_intersectshs2.C).all()
        assert(gt_intersect.C==out_intersectscale.C).all()
        assert(gt_intersect.C==out_intersectrotate.C).all()
        assert(gt_intersect.C==out_intersectopacity.C).all()

        shsloss = 4*MSEloss(gt_intersect.F[:,:3],-2*out_intersectshs.F)+4*MSEloss(gt_intersect.F[:,3:3+9],0.3*out_intersectshs2.F)
        #mseloss= shsloss+0.05*MSEloss(gt_intersect.F[:,12:15],0.003*out_intersectscale.F)+0.0001*MSEloss(torch.nn.functional.normalize(gt_intersect.F[:,15:19]),torch.nn.functional.normalize(out_intersectrotate.F))+0.02*MSEloss(gt_intersect.F[:,19:20], out_intersectopacity.F)
        renderlossd = 0
        accum_iter = 8  

        for samplej in range(len(batch_coords)):

            # rendershs = out6shs[samplej]
            rendershs = torch.concat([-2*out6shs[samplej], 0.3*out6shs2[samplej]],1)
            predptsnum = batch_coords[samplej].shape[0]
            renderpc = (batch_coords[samplej]/447)*torch.from_numpy(np.array(data_dict['std'][samplej])).cuda()+torch.from_numpy(np.array(data_dict['mean'][samplej])).cuda()
            
        # # print(renderpc.shape)
            if i %200 ==0:
            #     # print(renderpc.shape)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(renderpc.cpu().detach().numpy())
                o3d.io.write_point_cloud("recon_voxel.ply", pcd)
            rendershs = rendershs.reshape(predptsnum,4,3)
            renderrotation = torch.nn.functional.normalize(out6rotation[samplej])
            renderscale = out6scale[samplej]
            renderopacity = out6opacity[samplej]
       
        # rendering loss 

        
            shsgs = rendershs
        # # print(shsgs.shape)
        # # scalegs = torch.nn.Softplus()(renderscale)
            scalegs = 0.002*(renderscale)
            rotationgs = torch.nn.functional.normalize(renderrotation.unsqueeze(0))[0]
        # # print(rotationgs.shape)
            opacitygs = renderopacity

        # loss = 0
        # # pred_img = torch.tensor([]).cuda()
            # fanangle = (360- int(data_dict['angle'][samplej].zfill(3)))%360
            if len(data_dict['person'][samplej])==4:
                personname = data_dict['person'][samplej]
                fanangle = randint(1, 360-1)
                fanangle =  str(fanangle).zfill(3)
                radomangle = int(fanangle)
            else:
                personname = data_dict['person'][samplej][:4]
                fanangle = data_dict['person'][samplej][5:]
                fanangle =  str(fanangle).zfill(3)
                

                radomangle = int(fanangle)

            RTpath = os.path.join("/data/tangyingzhi/Project/gaussian-splatting/0000_512", "calibration"+str(radomangle).zfill(3)+".json")
            file = open(RTpath, 'r')
            js = file.read()
            dic = json.loads(js)
            RT = np.asarray(dic['RT']).reshape(3,4).transpose()*1000
            R = RT[:3]
            T = RT[-1]
            focallength = 711.111083984375
            height = 512
            width = 512
            FovY = focal2fov(focallength, height)
            FovX = FovY
            
            
            gtimg_path = os.path.join("/data/tangyingzhi/data/thuman21render/", personname,  "rendered_image_"+str(fanangle).zfill(3)+".png")
            
            imagegt = PILtoTorch2(Image.open(gtimg_path))[:3,...].cuda()
            mask = PILtoTorch2(Image.open(gtimg_path))[3:,...].cuda()
            zfar = 100.0
            znear = 0.01

            trans = [0.0, 0.0, 0.0]
            scaledd = 1.0

            world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scaledd)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovY, fovY=FovX).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).cuda()
            camera_center = world_view_transform.inverse()[3, :3].cuda()
            # Render
            # if (iteration - 1) == debug_from:
            #     pipe.debug = True
            random_background = False
            bg = torch.rand((3), device="cuda") if random_background else background
            
            if i>-1:
                render_pkg = render(renderpc, shsgs, scalegs, rotationgs, opacitygs,  world_view_transform, FovX, full_proj_transform, camera_center, bg)
                image, viewspace_point_tensor, visibility_filter, radii, colors_precomp = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['colors_precomp']
     
                # if i % 200==0:
                # if i % 200==0:
                
                cv2.imwrite("/data/tangyingzhi/Project/gaussian-splatting/testimage/all_"+personname+".png", cv2.cvtColor(image.permute(1,2,0).detach().cpu().numpy()*255, cv2.COLOR_RGB2BGR))
                cv2.imwrite("/data/tangyingzhi/Project/gaussian-splatting/testimage/all_"+personname+"gt.png", cv2.cvtColor(imagegt.permute(1,2,0).detach().cpu().numpy()*255, cv2.COLOR_RGB2BGR))
                

                Ll1 = l1_loss(image.cuda(), imagegt.cuda())
               

                bg = torch.rand((3), device="cuda") if random_background else background
                
                
                if i>-1:
                    renderloss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image , imagegt ))+0.02*loss_fnperc(image.cuda().squeeze().unsqueeze(0), imagegt.cuda().squeeze().unsqueeze(0))
                else: 
                    renderloss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image , imagegt ))
                renderlossd +=renderloss
            # occupancy loss 

        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        # print(BCE.grad_fn)
        # if i>1:
        #     loss = BCE+renderlossd+0.000001*kl_loss+mseloss
        # else:
        loss = BCE*2+0.000001*kl_loss+0.2*shsloss+renderlossd
        loss = loss / accum_iter 
        loss.backward()
        if ((i + 1) % accum_iter == 0):
        
            optimizer.step()
            optimizer.zero_grad()
        t = time() - s

        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.3e},  Depths: {len(out_cls)} Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
            )

        if i % 50000 == 0 and i > 0:
            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            net.train()
        if i % config.val_freq == 0 and i > 0:
            zssave = zs.dense()[0].detach().cpu().numpy()
            np.save("zssave.npy", zssave)

            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )

            



if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6030)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    pipe = pp.extract(args)
    

    

    net = AE()
    net.to(device)
    # logging.info(net)
    dataloader = make_data_loader(
        "train",
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config,
    )
    checkpoint = torch.load("/data/tangyingzhi/data/shsvaeresnet2.pth")
    net.load_state_dict(checkpoint["state_dict"])
    train(net, dataloader, device, config,pipe)
