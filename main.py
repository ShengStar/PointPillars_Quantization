from train import bulid_net
import torch
from predict import predict
import numpy as np
import time
def get_start_result_anno(): 
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations

def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations
    

if __name__ == '__main__':
    dt_annos=[]
    for n in range (1) :
        # i = str(n)
        # s = i.zfill(6)
        # box_preds = "/home8T/detection_head/result/" + s
        # cls_preds = "/home8T/detection_head/result/" + s
        # dir_cls_preds = "/home8T/detection_head/result/" + s
        # # box_preds = box_preds + "/"
        # # cls_preds = cls_preds + "/"
        # # dir_cls_preds = dir_cls_preds + "/"
        # box_preds = box_preds + "_box_preds.npy"
        # cls_preds = cls_preds + "_cls_preds.npy"
        # dir_cls_preds = dir_cls_preds + "_dir_cls_preds.npy"
        
        box_preds = torch.tensor(np.load("/home8T/detection_head/result/000001_box_preds.npy"))
        cls_preds = torch.tensor(np.load("/home8T/detection_head/result/000001_cls_preds.npy"))
        dir_cls_preds = torch.tensor(np.load("/home8T/detection_head/result/000001_dir_cls_preds.npy"))
        batch_anchor = torch.tensor(np.load("/home/star/rewrite_point_pillar_no_cuda/batch_anchors.npy"))
        print(box_preds.shape)
        print(cls_preds.shape)
        print(dir_cls_preds.shape)
        net = bulid_net()
        net.load_state_dict(torch.load("/home/star/pointpillars_res_1_save/second/data/model_202112201024/voxelnet-352640.tckpt"))
        net.eval()
        filename = '/home8T/000001.bin'
        preds_dict = net(filename)
        box_preds = torch.tensor(preds_dict["box_preds"])
        cls_preds = torch.tensor(preds_dict["cls_preds"])
        dir_cls_preds = torch.tensor(preds_dict["dir_cls_preds"])
        print(box_preds.shape)
        print(cls_preds.shape)
        print(dir_cls_preds.shape)
        image_idex = 1
        # box_preds = torch.tensor(np.load("/home/star/rewrite_point_pillar_no_cuda/batch_box_preds.npy"))
        # cls_preds = torch.tensor(np.load("/home/star/rewrite_point_pillar_no_cuda/batch_cls_preds.npy"))
        # dir_cls_preds = torch.tensor(np.load("/home/star/rewrite_point_pillar_no_cuda/batch_dir_preds.npy"))
        preds_dict = predict(batch_anchor,box_preds,cls_preds,dir_cls_preds,image_idex)
        box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
        label_preds = preds_dict["label_preds"].detach().cpu().numpy()
        scores = preds_dict["scores"].detach().cpu().numpy()
        box_preds = preds_dict["box3d_camera"]
        box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
        ## 保存
        annos = []
        anno = get_start_result_anno()
        class_names = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
            3: 'Van',
            4: 'Person_sitting',
        }
        num_example = 0
        # box_2d_preds = torch.tensor(np.load("/home/star/rewrite_point_pillar_no_cuda/box_2d_preds.npy"))
        center_limit_range = [0.0, -10.239999771118164, -5.0, 40.959999084472656, 10.239999771118164, 5.0]
        image_shape  = np.array([375,1242])
        for box,box_lidar,bbox,score,label in zip(box_preds,box_preds_lidar,box_2d_preds,scores,label_preds):
            if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                continue
            if bbox[2] < 0 or bbox[3] < 0:
                continue
            if center_limit_range is not None:
                limit_range = np.array(center_limit_range)
                if (np.any(box_lidar[:3] < limit_range[:3]) or np.any(box_lidar[:3] > limit_range[3:])):
                    continue
            bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
            bbox[:2] = np.maximum(bbox[:2], [0, 0])
            anno["name"].append(class_names[label])
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)
            anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
            anno["bbox"].append(bbox)
            anno["dimensions"].append(box[3:6])
            anno["location"].append(box[:3])
            anno["rotation_y"].append(box[6])
            anno["score"].append(score)
            num_example += 1
        if num_example != 0:
            anno = {n: np.stack(v) for n, v in anno.items()}
            annos.append(anno)
        else:
            annos.append(empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        img_idx = preds_dict["image_idx"]
        annos[-1]["image_idx"] = np.array([img_idx] * num_example, dtype=np.int64)
        dt_annos += annos
        print(annos)

    
