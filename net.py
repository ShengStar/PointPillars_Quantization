import torch
from torch import nn
import pickle

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.activation = nn.LeakyReLU(0.1)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        # begin res_block_1
        self.conv1_1 = BasicConv(64, 64, 3)
        self.conv1_2 = BasicConv(32, 32, 3)
        self.conv1_3 = BasicConv(32, 32, 3)
        self.conv1_4 = BasicConv(64, 64, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1_5 = BasicConv(128, 192, 1)
        # end res_block_1
        # begin res_block_2
        self.conv2_1 = BasicConv(128, 128, 3)
        self.conv2_2 = BasicConv(64, 64, 3)
        self.conv2_3 = BasicConv(64, 64, 3)
        self.conv2_4 = BasicConv(128, 128, 1)
        self.maxpool_1 = nn.MaxPool2d([2, 2], [2, 2])
        # end res_block_2
        # begin res_block_3
        self.conv3_1 = BasicConv(256, 256, 3)
        self.conv3_2 = BasicConv(128, 128, 3)
        self.conv3_3 = BasicConv(128, 128, 3)
        self.conv3_4 = BasicConv(256, 256, 1)
        #self.maxpool = nn.MaxPool2d([2, 2], [2, 2])
        # end res_block_3
        self.upsample = Upsample(512, 192)
        self.conv_cls = nn.Conv2d(384, 2, 1)
        self.conv_box = nn.Conv2d(384, 14, 1)
        self.conv_dir_cls = nn.Conv2d(384, 4, 1)
    def forward(self,x):
        x = self.conv1_1(x)
        route = x
        x = torch.split(x, 32, dim=1)[1]
        x = self.conv1_2(x)
        route1 = x
        x = self.conv1_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv1_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        feat1 = x
        feat12 = self.conv1_5(feat1)
        x = self.conv2_1(x)
        route = x
        x = torch.split(x, 64, dim=1)[1]
        x = self.conv2_2(x)
        route1 = x
        x = self.conv2_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv2_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool_1(x)
        x = self.conv3_1(x)
        route = x
        x = torch.split(x, 128, dim=1)[1]
        x = self.conv3_2(x)
        route1 = x
        x = self.conv3_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv3_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, feat12], axis=1)
        print("45654")
        print(x)


        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()  ##
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()  #
        dir_cls_preds = self.conv_dir_cls(x).permute(0, 2, 3, 1).contiguous()  ##
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
            "dir_cls_preds" : dir_cls_preds,
        }
        return ret_dict
