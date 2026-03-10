import torch
import MinkowskiEngine as ME
import torch.nn as nn 
import numpy as np 

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
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

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

def load_sparse_tensor(filedir, device):
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    feats = torch.ones((len(coords),1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    
    return x

def scale_sparse_tensor(x, factor):
    coords = (x.C[:,1:]*factor).round().int()
    feats = torch.ones((len(coords),1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    
    return x

criterion = torch.nn.BCEWithLogitsLoss()

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce
    
class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels):
        super().__init__()
        bn_momentum = 0.1
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels*2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.bn0_0 = ME.MinkowskiBatchNorm(channels*2, momentum=bn_momentum)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.bn0_1 = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels*2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.bn1_0 = ME.MinkowskiBatchNorm(channels*2, momentum=bn_momentum)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels*2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.bn1_1 = ME.MinkowskiBatchNorm(channels*2, momentum=bn_momentum)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.bn1_2 = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)
        self.elu = ME.MinkowskiELU()
        
    def forward(self, x):
        out0 = self.bn0_1(self.conv0_1(self.elu(self.bn0_0(self.conv0_0(x)))))
     
        out1 = self.bn1_2(self.conv1_2(self.elu(self.bn1_1(self.conv1_1(self.elu(self.bn1_0(self.conv1_0(x))))))))
        
        out = out0 + out1 + x

        return out

def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)

class Encoder(torch.nn.Module):
    def __init__(self, channels=[20,64,256,256,64,8]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=2, 
            channels=channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=2, 
            channels=channels[3])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[4])

        self.convmean = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.convvar = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)

        mean = self.convmean(out2)
        var = self.convvar(out2)

        return [out1, out0], mean, var


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """
    def __init__(self, channels=[8,64,128,128]):
        super().__init__()
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[1])

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2])

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3])

        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv2_shsdc = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=3,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv2_shsrest = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=9,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.conv2_scale = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=3,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv2_rotation = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.conv2_opacity = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        




        self.sigmoid = ME.MinkowskiSigmoid()
        self.tanh = ME.MinkowskiTanh()
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training: 
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else: 
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, nums_list, ground_truth_list, training=True):
        #
        out = self.relu(self.conv0(self.relu(self.up0(x))))
        out = self.block0(out)
        out_cls_0 = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls_0, 
            nums_list[0], ground_truth_list[0], training)
        #
        out = self.relu(self.conv1(self.relu(self.up1(out))))
        out = self.block1(out)
        out_cls_1 = self.conv1_cls(out)
        out = self.prune_voxel(out, out_cls_1, 
            nums_list[1], ground_truth_list[1], training)
        #
        out = self.relu(self.conv2(self.relu(self.up2(out))))
        out = self.block2(out)
        out_cls_2 = self.conv2_cls(out)
        shsdc = self.sigmoid(self.conv2_shsdc(out))
        shsrest = self.tanh(self.conv2_shsrest(out))
        scale = self.sigmoid(self.conv2_scale(out))
        rotation = ME.MinkowskiFunctional.normalize(self.conv2_rotation(out))
        opacity = self.sigmoid(self.conv2_opacity(out))
        out = ME.cat(shsdc, shsrest, scale, rotation, opacity )
        out = self.prune_voxel(out, out_cls_2, 
            nums_list[2], ground_truth_list[2], training)
        # shsdc = self.prune_voxel(shsdc, out_cls_2, 
        #     nums_list[2], ground_truth_list[2], training)
        # shsrest = self.prune_voxel(shsrest, out_cls_2, 
        #     nums_list[2], ground_truth_list[2], training)
        # scale = self.prune_voxel(scale, out_cls_2, 
        #     nums_list[2], ground_truth_list[2], training)
        # rotation = self.prune_voxel(rotation, out_cls_2, 
        #     nums_list[2], ground_truth_list[2], training)
        # opacity = self.prune_voxel(opacity, out_cls_2, 
        #     nums_list[2], ground_truth_list[2], training)

        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]
        
        return out_cls_list, out

class AE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.quantizer = VectorQuantizer(n_e=2048, e_dim=8, beta=0.25)
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar.F)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, sinput):
        # print(sinput.shape)
        y_list, means, var = self.encoder(sinput)
        zs = self.reparameterize(means, var)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + var.F - means.F.pow(2) - var.F.exp(),1))
        ground_truth_list = y_list + [sinput] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        out_cls_list, out = self.decoder(zs, nums_list, ground_truth_list, True)

        return out_cls_list, ground_truth_list,  out,   kl_loss