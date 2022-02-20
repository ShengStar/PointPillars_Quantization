# SSD检测头
import torch
import numpy as np
from core import box_torch_ops
from torch import stack as tstack

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([
            tstack([rot_cos, zeros, -rot_sin]),
            tstack([zeros, ones, zeros]),
            tstack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([
            tstack([rot_cos, -rot_sin, zeros]),
            tstack([rot_sin, rot_cos, zeros]),
            tstack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tstack([
            tstack([zeros, rot_cos, -rot_sin]),
            tstack([zeros, rot_sin, rot_cos]),
            tstack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))

def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = np.float32
    # print("dtype123")
    # print(dims)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    # print("dims")
    # print(dims)
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners

def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res

def lidar_to_camera(points, r_rect, velo2cam):
    # # print(points)
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    
    camera_points = points @ (np.array(r_rect) @ np.array(velo2cam)).T
    return camera_points[..., :3]

def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    # # print(xyz_lidar)
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)

def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1) # 撕anchor
    xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1) # 撕特征图

    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa

    yg = yt * diagonal + ya
    zg = zt * ha + za
    # print(lt)
    lg = np.exp(lt) * la
    # print(lg)
    wg = np.exp(wt) * wa
    hg = np.exp(ht) * ha

    rg = rt + ra
    zg = zg - hg / 2
    
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def predict(batch_anchors,batch_box_preds,batch_cls_preds,batch_dir_preds,image_idex):
    batch_size = 1 #*
    
    nms_score_threshold = 0.05
    num_class_with_bg = 1
    batch_anchors_mask = [None] * batch_size
    # batch_anchors_mask = anchors_mask.view(batch_size, -1)
    batch_box_preds = batch_box_preds.contiguous().view(1,-1,7)
    batch_cls_preds = batch_cls_preds.contiguous().view(1,-1, 1)
    batch_box_preds = second_box_decode(batch_box_preds,batch_anchors)
    batch_dir_preds = batch_dir_preds.contiguous().view(batch_size, -1, 2)

    
    rect =[ [ 0.9999,  0.0098, -0.0074,  0.0000],
        [-0.0099,  0.9999, -0.0043,  0.0000],
        [ 0.0074,  0.0044,  1.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    Trv2c = [[ 7.5337e-03, -9.9997e-01, -6.1660e-04, -4.0698e-03],
        [ 1.4802e-02,  7.2807e-04, -9.9989e-01, -7.6316e-02],
        [ 9.9986e-01,  7.5238e-03,  1.4808e-02, -2.7178e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    for box_preds, cls_preds, dir_preds,a_mask in zip(batch_box_preds,batch_cls_preds,batch_dir_preds,batch_anchors_mask):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
        dir_labels = torch.max(dir_preds, dim=-1)[1]
        total_scores = torch.sigmoid(cls_preds)
        nms_func = box_torch_ops.nms
        selected_boxes = None
        selected_labels = None
        selected_scores = None
        selected_dir_labels = None
        if num_class_with_bg == 1:
            top_scores = total_scores.squeeze(-1)
            top_labels = torch.zeros(total_scores.shape[0],device=total_scores.device,dtype=torch.long)
        if nms_score_threshold >0.0:
            thresh = torch.tensor([nms_score_threshold],device=total_scores.device).type_as(total_scores)
            top_scores_keep = (top_scores >= thresh)
            top_scores = top_scores.masked_select(top_scores_keep).cuda()
        if top_scores.shape[0] != 0:
            if nms_score_threshold > 0.0:
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                top_labels = top_labels[top_scores_keep]
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            box_preds_corners = box_torch_ops.center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],boxes_for_nms[:, 4])
            boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners).cuda()
            # the nms in 3d detection just remove overlap boxes.
            # print("haha")
            # print(boxes_for_nms)
            # print(top_scores)
            selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=1000,
                        post_max_size=300,
                        iou_threshold=0.5,
                    )
            # print("hahaha")
            # print(selected.shape)
        else:
            selected = None
        if selected is not None:
            
            selected_boxes = box_preds[selected]
            # print(selected.shape)
            # print("box_2d_preds7")
            # print(selected_boxes.shape)
            # print(selected_boxes)

            selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
        # finally generate predictions.
        # 最终生成预测
        if selected_boxes is not None:
            box_preds = selected_boxes
            # print("box_2d_preds6")
            # print(box_preds.shape)
            # print(box_preds)
            scores = selected_scores
            label_preds = selected_labels
            dir_labels = selected_dir_labels
            opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
            # print("box_2d_preds5")
            # print(box_preds.shape)
            box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
            
            final_box_preds = box_preds
            # print("box_2d_preds4")
            # print(final_box_preds.shape)
            final_scores = scores
            final_labels = label_preds
            final_box_preds_camera = box_lidar_to_camera(final_box_preds, rect, Trv2c)
            final_box_preds_camera = torch.from_numpy(final_box_preds_camera)
            # print("box_2d_preds3")
            # print(final_box_preds_camera.shape)
            ##bbox
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=1)
            # print("box_2d_preds2")
            # print(box_corners.shape)
            P2 = [[7.2154e+02, 0.0000e+00, 6.0956e+02, 4.4857e+01],
            [0.0000e+00, 7.2154e+02, 1.7285e+02, 2.1638e-01],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-03],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]
            P2= torch.from_numpy(np.array(P2))
            box_corners_in_image = project_to_image(box_corners, P2)
            # print("box_2d_preds1")
            # print(box_corners_in_image.shape)
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            # print("box_2d_preds")
            # print(minxy.shape)
            # print(maxxy.shape)
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            ##bbox
            predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": image_idex,
                }
        else:
            predictions_dict = {
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                }
        return predictions_dict