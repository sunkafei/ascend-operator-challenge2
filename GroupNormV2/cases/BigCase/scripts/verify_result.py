import os
import sys
import numpy as np

loss = 1e-4 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):
    mean = np.fromfile("/home/lhq/ascend-operator-challenge2/GroupNormV2/cases/BigCase/output/mean.bin", dtype=np.float32)
    rstd = np.fromfile("/home/lhq/ascend-operator-challenge2/GroupNormV2/cases/BigCase/output/rstd.bin", dtype=np.float32)
    print("mean:", mean, file=sys.stderr) # mean: [ 0.00106605 -0.00164301]
    print("rstd:", rstd, file=sys.stderr) # rstd: [5.772267  5.7724257]
    real_result = np.fromfile(real_result, dtype=np.float32) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.float32) # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("golden:", golden[:8], file=sys.stderr)
            print("real_result:", real_result[:8], file=sys.stderr)
            print(np.sum(result_rtol == False), np.sum(result_atol == False), file=sys.stderr)
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
