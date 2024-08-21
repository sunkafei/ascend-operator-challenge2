import torch
import torch.nn as nn
import numpy as np
import os

np.random.seed(123)
def gen_golden_data_simple():
    input_x = np.random.uniform(-10000, 10000, [16, 16, 64]).astype(np.float32)
    input_gamma = np.random.uniform(1, 1, [16]).astype(np.float32)
    input_beta = np.random.uniform(0, 0, [16]).astype(np.float32)
    num_groups = np.array([4]).astype(np.int32)
    data_format = "NCHW"
    eps = 0.5
    is_training = True
    res = torch.nn.functional.group_norm(torch.Tensor(input_x), num_groups=num_groups[0],eps=eps)
    golden = res.numpy().astype(np.float32)
    print("golden = ", golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_gamma.tofile("./input/input_gamma.bin")
    input_beta.tofile("./input/input_beta.bin")
    golden.tofile("./output/golden.bin")
    print("mean:", np.mean(input_x.reshape(input_x.shape[0], -1), axis=1))
    print("std2:", np.std(input_x.reshape(input_x.shape[0], -1), axis=1) ** 2)


if __name__ == "__main__":
    gen_golden_data_simple()
