from net import backbone
from process import read_point,pointpillars,voxel_feature_extractor,Scatter
import torch
from torch import nn
import torchplus
from torchplus import metrics
import numpy as np

def sparse_sum_for_anchors_mask(coors, shape):
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 0]] += 1
    return ret

def fused_get_anchors_area(dense_map, anchors_bv, stride, offset,
                           grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros((N), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor(
            (anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor(
            (anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor(
            (anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor(
            (anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
    return ret

class bulid_net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=True)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=True,
            encode_background_as_zeros=True)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self.voxel_feature_extractor = voxel_feature_extractor()
        self.rpn = backbone()
    

    def forward(self, filename):
        readpoint = read_point()
        x = readpoint(filename)  # 读取点云
        pointpillar = pointpillars()
        voxels, num_points_per_voxel, coors = pointpillar(x)  # 划分pillars
        print("!!!!!!")
        print(voxels)
        print(num_points_per_voxel)
        print(coors)
        print(num_points_per_voxel.shape)
        features_9 = self.voxel_feature_extractor(voxels, num_points_per_voxel, coors)  # 特征拓展
        scatter = Scatter()
        x = scatter(features_9, coors, 1)
        print("12321")
        print(x.shape)
        ret_dict =self.rpn(x)

        # 创建 anchor_mask
        # anchors_mask = None
        # # (496,432)
        # # coors,xyz
        # dense_voxel_map = sparse_sum_for_anchors_mask(coors,[496,432])
        # dense_voxel_map = dense_voxel_map.cumsum(0)
        # dense_voxel_map = dense_voxel_map.cumsum(1)
        # anchors_bv = np.load("files/anchors_bv.npy")
        # [0.16,0.16,4]
        # [0,-39.68,-3,69.12,39.68,1]
        # [432,496,1]
        # anchors_area = fused_get_anchors_area(dense_voxel_map,anchors_bv,[0.16,0.16,4],[0,-39.68,-3,69.12,39.68,1],[432,496,1])
        # anchors_mask = anchors_area > anchor_area_threshold
        # anchors_mask = anchors_mask.astype(np.uint8)
        # m = 0
        # for i in range(107136):
        #     m = m + anchors_mask[i]
        #     print(m)
        # print(sum(anchors_mask))
        return ret_dict
