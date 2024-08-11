import os
import sys
import numpy as np

loss = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一


def verify_result(real_result, golden):
    b = 5
    m = 20
    sample_num = 2
    
    real_result = np.fromfile(real_result, dtype=np.int32) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.int32) # 从bin文件读取预期运算结果
    all_len = len(golden)
    real_result = np.reshape(real_result,[b, m ,sample_num])
    golden = np.reshape(golden,[b, m ,sample_num])
    diff_num = 0 
    for i in range(b):
        for m in range(m):
            real_tmp = np.unique(real_result[i][m])
            golden_tmp = np.unique(golden[i][m])
            mask = ~np.in1d(real_tmp, golden_tmp)
            diff_num = diff_num + len(real_tmp[mask])

    if diff_num < all_len * loss:
        print("test pass")
        return True
    else:
        print("[ERROR] result error")
        return False


if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
