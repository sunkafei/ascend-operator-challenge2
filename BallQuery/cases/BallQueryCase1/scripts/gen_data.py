#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import torch


def ball_query(xyz, center_xyz, min_radius, max_radius, sample_num):
    b, m, _ = center_xyz.shape
    b, n, _ = xyz.shape
    res = np.zeros([b, m, sample_num], dtype=np.int32)
    for i in range(b):
        for j in range(m):
            center_x = center_xyz[i][j][0]
            center_y = center_xyz[i][j][1]
            center_z = center_xyz[i][j][2]
            cnt = 0

            for k in range(n):
                x = xyz[i][k][0]
                y = xyz[i][k][1]
                z = xyz[i][k][2]
                dis = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2 + (center_z - z) ** 2)
                if dis == 0 or min_radius <= dis < max_radius:
                    if cnt == 0:
                        for t in range(sample_num):
                            res[i][j][t] = k
                    res[i][j][cnt] = k
                    cnt += 1
                    if cnt >= sample_num:
                        break

    return np.array(res)


def stack_ball_query(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, max_radius, sample_num):
    radius2 = max_radius * max_radius
    center_xyz_length = center_xyz.shape[0]
    idx = torch.zeros((center_xyz_length, sample_num), dtype=torch.int32)
    for i in range(center_xyz_length):
        current_b_idx = 0
        tmp_b = 0
        for _b in range(len(center_xyz_batch_cnt)):
            tmp_b += center_xyz_batch_cnt[_b]
            if tmp_b > i:
                current_b_idx = _b
                break
        new_x = center_xyz[i][0]
        new_y = center_xyz[i][1]
        new_z = center_xyz[i][2]
        n = xyz_batch_cnt[current_b_idx]

        xyz_offset = 0

        for _t in range(current_b_idx):
            xyz_offset += xyz_batch_cnt[_t]

        cnt = 0
        for j in range(n):
            x = xyz[xyz_offset + j][0]
            y = xyz[xyz_offset + j][1]
            z = xyz[xyz_offset + j][2]
            dis = (new_x - x) ** 2 + (new_y - y) ** 2 + (new_z - z) ** 2
            if dis < radius2:
                if cnt == 0:
                    for f in range(sample_num):
                        idx[i][f] = j
                idx[i][cnt] = j
                cnt += 1
                if cnt >= sample_num:
                    break
    return idx.numpy()


def gen_golden_data_simple():
    b = 5
    m = 64
    n = 32

    input_xyz = np.random.uniform(-10, 10, [b, n, 3]).astype(np.float16)
    center_xyz = np.random.uniform(-10, 10, [b, m, 3]).astype(np.float16)
    xyz_batch_cnt = None
    center_xyz_batch_cnt = None
    min_radius = 10
    max_radius = 50
    sample_num = 17
    if xyz_batch_cnt is None:
        golden = ball_query(input_xyz, center_xyz, min_radius, max_radius, sample_num)
    else:
        golden = stack_ball_query(input_xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, max_radius, sample_num)


    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_xyz.tofile("./input/input_xyz.bin")
    center_xyz.tofile("./input/center_xyz.bin")
    golden.tofile("./output/golden.bin")
    if xyz_batch_cnt is not None:
        xyz_batch_cnt.tofile("./input/xyz_batch_cnt.bin")
        center_xyz_batch_cnt.tofile("./input/center_xyz_batch_cnt.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
