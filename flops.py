import torch
# from nets.CV-Main import CV-Main as create_model
# from nets.MySeg_Model import MySegNet as create_model
# from nets.segnet import SegNet as create_model
from model.UNetPlus import UnetPlusPlus as create_model
from calflops import calculate_flops

model = create_model(3, 2)
flops, macs, params = calculate_flops(model, input_shape=(1, 3, 128, 128))
print(flops, params)