import torch
from torch import nn
import numpy as np
import inspect
from torch.nn import functional as F



class Scatter(nn.Module):##把图像恢复为伪图
    def __init__(self,
                 output_shape=[1, 1, 128, 256, 64],
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size = 1):
        print(voxel_features.shape)
        print(coords.shape)
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,device=voxel_features.device)##[64,400*352]
            print("canvas1")
            print(canvas.shape)
            # Only include non-empty pillars 仅包含非空的pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            print("canvas2")
            print(this_coords.shape)
            print(batch_mask.shape)
            print(coords.shape)
            indices = this_coords[:, 1] * self.nx + this_coords[:, 2]
            #indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels
            # Append to a list for later stacking.
            batch_canvas.append(canvas)
        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)
        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)##[1,64,400,352] 很可能出错的位置
        print("batch_canvas")
        print(batch_canvas)
        return batch_canvas

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class voxel_feature_extractor(nn.Module):
    def __init__(self):
        super(voxel_feature_extractor, self).__init__()
        num_input_features = 9
        num_filters = (64,)
        use_norm = True
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        voxel_size = (0.16, 0.16, 4)
        pc_range = (0, -10.24, -3, 40.96, 10.24, 1)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
    def forward(self, features, num_voxels, coors):
        features = torch.from_numpy(features)##把特征转换为numpy
        num_voxels = torch.from_numpy(num_voxels)##每个voxel中的点
        coors = torch.from_numpy(coors)##voxel的坐标
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.view(-1, 1, 1)##points_mean:大小[11760,1,3]
        f_cluster = features[:, :, :3] - points_mean ##features[:, :, :3]:[11760,100,3]##每个值与中心值的差f_cluster[11760,100,3]
        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])##features[:, :, :2] [11760,100,2]
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 2].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * self.vy + self.y_offset)
        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        features = torch.cat(features_ls, dim=-1)##【11760,100,9】
        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)##特征拓展
        print("features!!")
        print(features)
        return features.squeeze()

class read_point(nn.Module):
    def __init__(self):
        super(read_point, self).__init__()
    def forward(self, x,):
        filename = x
        pointcloud = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1, 4])
        return pointcloud
class pointpillars(nn.Module):
    def __init__(self):
        super(pointpillars, self).__init__()
        self.coors_range = np.array([0, -10.24, -3, 40.96, 10.24, 1])
        self.voxel_size = np.array([0.16, 0.16, 4])

    def forward(self, x,):
        max_points = 100
        max_voxels = 12000
        coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)#创建坐标存储矩阵 [12000,3]
        num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)#创建每个pillar中的点云计数[12000]
        voxels = np.zeros(shape=(max_voxels, max_points, x.shape[-1]), dtype=x.dtype)##创建voxel索引列表，[12000,100,4]
        N = x.shape[0]#点云的数量
        ndim = 3
        ndim_minus_1 = ndim - 1
        grid_size = (self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size ##(352,400,1)
        voxelmap_shape = (self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())##(352,400,1)
        voxelmap_shape = voxelmap_shape[::-1]
        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)##[352,400,1]的全-1矩阵
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)##[352,400,1]
        coor = np.zeros(shape=(3,), dtype=np.int32) ##(0 0 0)
        voxel_num = 0
        failed = False
        print(x)
        for i in range(N):
            failed = False
            for j in range(ndim):
                c = np.floor((x[i, j] - self.coors_range[j]) / self.voxel_size[j])
                if c < 0 or c >= grid_size[j]:
                    failed = True
                    break
                coor[ndim_minus_1 - j] = c
            if failed:
                continue
            voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]##
            if voxelidx == -1:
                voxelidx = voxel_num
                if voxel_num >= max_voxels:
                    break
                voxel_num += 1
                coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
                coors[voxelidx] = coor
            num = num_points_per_voxel[voxelidx]
            if num < max_points:
                voxels[voxelidx, num] = x[i]
                num_points_per_voxel[voxelidx] += 1
        print("voxel_num")
        print(voxel_num)
        coors = coors[:voxel_num]##每个voxel的坐标
        voxels = voxels[:voxel_num]##voxel和其中的点
        num_points_per_voxel = num_points_per_voxel[:voxel_num]#每个voxel中的点的数量
        print("pointpillars")
        print(voxels.shape)
        print(num_points_per_voxel.shape)
        print(coors.shape)
        return voxels, num_points_per_voxel, coors