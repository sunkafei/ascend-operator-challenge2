import torch
import torch.nn as nn
import numpy as np
import os

np.random.seed(456)
def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [3, 4, 32]).astype(np.float16)
    input_gamma = np.random.uniform(1, 1, [4]).astype(np.float16)
    input_beta = np.random.uniform(0, 0, [4]).astype(np.float16)
    num_groups = np.array([2]).astype(np.int32)
    data_format = "NCHW"
    eps = 0.0001
    is_training = True
    res = torch.nn.functional.group_norm(torch.Tensor(input_x), num_groups=num_groups[0],eps=eps)
    golden = res.numpy().astype(np.float16)
    print("golden = ", golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_gamma.tofile("./input/input_gamma.bin")
    input_beta.tofile("./input/input_beta.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
