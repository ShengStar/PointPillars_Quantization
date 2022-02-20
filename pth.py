import torch
chpt = torch.load('voxelnet-296960.pth')
# epoch = chpt['epoch']
# loss  = chpt['loss']
for k, v in chpt.items():
  print(k)