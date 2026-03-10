import os
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
import math
import MinkowskiEngine as ME


##############################################
def array2vector(array, step=None):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().clone(), step.long().clone()
    if array.min()<0:
        min_value = array.min()
        array = array - min_value
        step = step - min_value
        
    assert array.min()>=0 and array.max()-array.min()<step
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    data = data.clone()
    ground_truth = ground_truth.clone()
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    # positive value
    min_value =  torch.min(data.min(), ground_truth.min())
    if min_value < 0:
        data[:,1:] -= min_value
        ground_truth[:,1:] -= min_value
    #
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask

def istopk_local(data, k=1):
    """input data is probability
        select top-k voxels in each 8-voxels set
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.reshape(-1,8), k)
    indices += (torch.arange(0, len(indices))*8).reshape(-1,1).to(indices.device)
    indices = indices.reshape(-1)
    mask[indices] = True
    
    return mask.bool().to(data.device)

def istopk_global(data, k):
    """input data is probability
        select top-k voxel in all voxels
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.squeeze(), k)
    mask[indices] = True

    return mask.bool().to(data.device)



bce_fn = torch.nn.BCEWithLogitsLoss()
ce_fn = torch.nn.CrossEntropyLoss()
softmax_fn = torch.nn.Softmax(dim=-1)


def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    if data.shape[-1]==8:
        assert groud_truth.F.shape[-1]==8
        bce = bce_fn(data.F, groud_truth.F)
    elif data.F.shape[-1]==1:
        assert groud_truth.F.shape[-1]==1
        if len(data)==len(groud_truth):
            bce = bce_fn(data.F.squeeze(), groud_truth.F.squeeze())
        else:
            mask = isin(data.C, groud_truth.C)
            bce = bce_fn(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0] * groud_truth.shape[1]
    
    return sum_bce


def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits


def sort_sparse_tensor(sparse_tensor, target=None):
    """ Sort points in sparse tensor according to their coordinates or the coords of target
    """
    if target is not None and (sparse_tensor.C==target.C).all():
        return ME.SparseTensor(features=sparse_tensor.F, 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    # positive value
    coords = sparse_tensor.C.clone()
    min_value =  coords.min()
    if min_value < 0: coords[:,1:] -= min_value
    # sort
    indices = torch.argsort(array2vector(coords, coords.max()+1)).cpu()
    out_coords = sparse_tensor.C[indices]
    # print('DBG!!!device:\t', indices.device)
    # print('DBG!!!device:\t', sparse_tensor.F.cpu().device)
    out_feats = sparse_tensor.F.cpu()[indices]
    out = ME.SparseTensor(coordinates=out_coords, 
                        features=out_feats, 
                        tensor_stride=sparse_tensor.tensor_stride, 
                        device=sparse_tensor.device)

    if target is not None:
        # positive value
        target_coords = target.C.clone()
        min_value =  target_coords.min()
        if min_value < 0: target_coords[:,1:] -= min_value
        # sort
        target_indices = torch.argsort(array2vector(target_coords, target_coords.max()+1))
        inverse_indices = target_indices.sort()[1].cpu()
        assert (out_coords[inverse_indices]==target.C).all()
        out = ME.SparseTensor(features=out_feats[inverse_indices], 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    return out


def dense2sparse(tensor):
    """convert a dense tensor to a sparse tensor
    """
    shape = tensor.shape

    ## method 1
    # all_ones = torch.ones(shape[:-1])
    # coordinates = torch.stack(torch.where(all_ones))
    # coordinates = torch.transpose(coordinates,0,1)

    ## method 2
    indices_list = [torch.arange(s) for s in shape[:-1]]
    meshgrids = torch.meshgrid(*indices_list, indexing='ij')
    coordinates = torch.stack([grid.flatten() for grid in meshgrids], dim=1)

    features = tensor.reshape(-1,shape[-1])

    return coordinates, features


##################################################
pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
def downscale_coords_ME(coords, scale):
    """downscale using pooling
    """
    feats = np.zeros((len(coords), 1))
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords, 
                                    tensor_stride=1, device='cpu')
    for idx in range(scale):
        sparse_tensor = pooling(sparse_tensor)
    coords = sparse_tensor.C.cpu().numpy()
    coords = coords[:,1:] // sparse_tensor.tensor_stride[0]
    
    return coords

from utils_sparse import array2vector
    
def downscale_coords(coords, scale):
    out_coords = coords.copy()

    for idx in range(scale):
        out_coords = np.floor(out_coords/2).astype('int')
        out_coords = np.unique(out_coords, axis=0)

    if False:
        out_coords1 = downscale_coords_ME(coords, scale)
        out_coords1 = torch.tensor(out_coords1)
        indices1 = torch.argsort(array2vector(out_coords1, out_coords1.max()+1)).cpu()
        out_coords1 = out_coords1[indices1]

        out_coords2 = torch.tensor(out_coords)
        indices2 = torch.argsort(array2vector(out_coords2, out_coords2.max()+1)).cpu()
        out_coords2 = out_coords2[indices2]
        
        assert (out_coords1==out_coords2).all()
        
    return out_coords