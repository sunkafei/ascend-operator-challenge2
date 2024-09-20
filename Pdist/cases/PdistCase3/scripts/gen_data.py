import torch
import numpy as np
import os


def gen_golden_data_simple():
    test_type = np.float32
    input_x = np.random.uniform(-1000, 1000, [224,224]).astype(test_type)
    p = 4.0
    res = torch.nn.functional.pdist(torch.from_numpy(input_x.astype(np.float32)), p)

    golden = res.numpy().astype(test_type)
    print("golden", golden)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()