import torch
import torch.nn as nn
import numpy as np
import os

def depth_to_space_forward(x, block_size, mode='DCR', data_format='NHWC'):
    input_data = x
    if mode == 'DCR' and data_format == 'NCHW':
        b, c, h, w = x.shape
        tmp = np.reshape(input_data, [b, block_size, block_size, c//(block_size**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        res = np.reshape(tmp, [b, c//(block_size ** 2), h * block_size, w * block_size])
    elif mode == 'CRD' and data_format == 'NCHW':
        b, c, h, w = x.shape
        tmp = np.reshape(input_data, [b, c//(block_size**2), block_size, block_size, h, w])
        tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        res = np.reshape(tmp, [b, c//(block_size ** 2), h * block_size, w * block_size])
    elif mode == 'DCR' and data_format == 'NHWC':
        b, h, w, c = x.shape
        tmp = np.reshape(input_data, [b, h, w, block_size, block_size, c//(block_size**2)])
        tmp = np.transpose(tmp, [0, 1, 3, 2, 4, 5])
        res = np.reshape(tmp, [b, h * block_size, w * block_size, c//(block_size ** 2)])
    else :
        b, h, w, c = x.shape
        tmp = np.reshape(input_data, [b, h, w, c//(block_size**2), block_size, block_size])
        tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        res = np.reshape(tmp, [b, h * block_size, w * block_size, c//(block_size ** 2)])
    return res

def gen_golden_data_simple():
    input_x = np.random.uniform(-5, 5, [3, 32, 32, 4]).astype(np.float16)

    block_size = 2
    mode = "DCR"
    data_format="NHWC"
    golden = depth_to_space_forward(input_x,block_size,mode,data_format)
    print(golden.shape)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
