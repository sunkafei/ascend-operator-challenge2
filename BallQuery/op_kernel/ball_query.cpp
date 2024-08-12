#include "kernel_operator.h"
using namespace AscendC;

template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR center, GM_ADDR points, GM_ADDR indices, float min_radius, float max_radius, int32_t sample_num, int32_t batch_size, int32_t num_points, int32_t num_centers) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->min_radius = min_radius;
        this->max_radius = max_radius;
        this->sample_num = sample_num;
        this->batch_size = batch_size;
        this->num_points = num_points;
        this->num_centers = num_centers;

        pointsGm.SetGlobalBuffer((__gm__ T*)points, batch_size * num_points * 3);
        centerGm.SetGlobalBuffer((__gm__ T*)center, batch_size * num_centers * 3);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, batch_size * num_centers * sample_num);
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_centers; ++j) {
                float center_x = centerGm.GetValue(i * num_centers * 3 + j * 3 + 0);
                float center_y = centerGm.GetValue(i * num_centers * 3 + j * 3 + 1);
                float center_z = centerGm.GetValue(i * num_centers * 3 + j * 3 + 2);
                int32_t cnt = 0;
                for (int k = 0; k < num_points; ++k) {
                    float x = pointsGm.GetValue(i * num_points * 3 + k * 3 + 0);
                    float y = pointsGm.GetValue(i * num_points * 3 + k * 3 + 1);
                    float z = pointsGm.GetValue(i * num_points * 3 + k * 3 + 2);
                    float dis = sqrt((center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) + (center_z - z) * (center_z - z));
                    if (dis == 0 || (min_radius <= dis && dis < max_radius)) {
                        if (cnt == 0) {
                            for (int t = 0; t < sample_num; ++t) {
                                indicesGm.SetValue(i * num_centers * sample_num + j * sample_num + t, k);
                            }
                        }
                        indicesGm.SetValue(i * num_centers * sample_num + j * sample_num + cnt, k);
                        cnt += 1;
                        if (cnt >= sample_num) {
                            break;
                        }
                    }
                }
            }
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(indicesGm);
    }

private:
    GlobalTensor<T> pointsGm, centerGm;
    GlobalTensor<int32_t> indicesGm;
    float min_radius, max_radius;
    int32_t sample_num, batch_size, num_points, num_centers;
};
template<typename T> class BruteForceStack {
public:
    __aicore__ inline BruteForceStack() {}
    __aicore__ inline void Init(GM_ADDR center, GM_ADDR points, GM_ADDR center_batch_cnt, GM_ADDR points_batch_cnt, GM_ADDR indices, float min_radius, float max_radius, int32_t sample_num, int32_t batch_size, int32_t num_points, int32_t num_centers) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->min_radius = min_radius;
        this->max_radius = max_radius;
        this->sample_num = sample_num;
        this->batch_size = batch_size;

        pointsGm.SetGlobalBuffer((__gm__ T*)points, num_points * 3);
        centerGm.SetGlobalBuffer((__gm__ T*)center, num_centers * 3);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, num_centers * sample_num);
        center_batch_cntGm.SetGlobalBuffer((__gm__ int32_t*)center_batch_cnt, batch_size);
        points_batch_cntGm.SetGlobalBuffer((__gm__ int32_t*)points_batch_cnt, batch_size);
    }
    __aicore__ inline void Process() {
        auto radius2 = max_radius * max_radius;
        // center_xyz_length = num_centers
        for (int i = 0; i < num_centers; ++i) {
            int current_b_idx = 0;
            int tmp_b = 0;
            for (int _b = 0; _b < batch_size; ++_b) {
                tmp_b += center_batch_cntGm.GetValue(_b);
                if (tmp_b > i) {
                    current_b_idx = _b;
                    break;
                }
            }
            float new_x = centerGm.GetValue(i * num_centers + 0);
            float new_y = centerGm.GetValue(i * num_centers + 1);
            float new_z = centerGm.GetValue(i * num_centers + 2);
            int n = points_batch_cntGm.GetValue(current_b_idx);

            int xyz_offset = 0;

            for (int _t = 0; _t < current_b_idx; ++_t) {
                xyz_offset += points_batch_cntGm.GetValue(_t);
            }

            int cnt = 0;
            for (int j = 0; j < n; ++j) {
                float x = pointsGm.GetValue((xyz_offset + j) * 3 + 0);
                float y = pointsGm.GetValue((xyz_offset + j) * 3 + 1);
                float z = pointsGm.GetValue((xyz_offset + j) * 3 + 2);
                float dis = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
                if (dis < radius2) {
                    if (cnt == 0) {
                        for (int f = 0; f < sample_num; ++f) {
                            indicesGm.SetValue(i * sample_num + f, j);
                        }
                    }
                    indicesGm.SetValue(i * sample_num + cnt, j);
                    cnt += 1;
                    if (cnt >= sample_num) {
                        break;
                    }
                }
            }
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(indicesGm);
    }

private:
    GlobalTensor<T> pointsGm, centerGm;
    GlobalTensor<int32_t> indicesGm, center_batch_cntGm, points_batch_cntGm;
    float min_radius, max_radius;
    int32_t sample_num, batch_size, num_points, num_centers;
};
template<typename T> class BallQuery {
public:
    __aicore__ inline BallQuery() {}
    __aicore__ inline void Init(GM_ADDR center, GM_ADDR points, GM_ADDR indices, float min_radius, float max_radius, int32_t sample_num, int32_t batch_size, int32_t num_points, int32_t num_centers) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->min_radius = min_radius * min_radius;
        this->max_radius = max_radius * max_radius;
        this->sample_num = sample_num;
        this->batch_size = batch_size;
        this->num_points = num_points;
        this->num_centers = num_centers;
        int cores = GetBlockNum();
        int span = (batch_size * num_centers - 1) / cores + 1;
        this->L = span * GetBlockIdx();
        this->R = span * (GetBlockIdx() + 1);
        if (this->R > batch_size * num_centers) {
            this->R = batch_size * num_centers;
        }

        pointsGm.SetGlobalBuffer((__gm__ T*)points, batch_size * num_points * 3);
        centerGm.SetGlobalBuffer((__gm__ T*)center, batch_size * num_centers * 3);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, batch_size * num_centers * sample_num);
    }
    __aicore__ inline void Process() { // 8350
        for (int i = L; i < R; ++i) {
            float center_x = centerGm.GetValue(i * 3 + 0);
            float center_y = centerGm.GetValue(i * 3 + 1);
            float center_z = centerGm.GetValue(i * 3 + 2);
            int32_t cnt = 0;
            const int batch = i / num_centers * num_points;
            for (int k = 0; k < num_points; ++k) {
                float x = pointsGm.GetValue((batch + k) * 3 + 0);
                float y = pointsGm.GetValue((batch + k) * 3 + 1);
                float z = pointsGm.GetValue((batch + k) * 3 + 2);
                float dis = (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) + (center_z - z) * (center_z - z);
                if (dis == 0 || (min_radius <= dis && dis < max_radius)) {
                    indicesGm.SetValue(i * sample_num + cnt, k);
                    cnt += 1;
                    if (cnt >= sample_num) {
                        break;
                    }
                }
            }
            if (cnt > 0 && cnt < sample_num) {
                int k = indicesGm.GetValue(i * sample_num + cnt - 1);
                while (cnt < sample_num) {
                    indicesGm.SetValue(i * sample_num + cnt, k);
                    cnt += 1;
                }
            }
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(indicesGm);
    }

private:
    GlobalTensor<T> pointsGm, centerGm;
    GlobalTensor<int32_t> indicesGm;
    float min_radius, max_radius;
    int32_t sample_num, batch_size, num_points, num_centers;
    int32_t L, R;
};
extern "C" __global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.type == -1) {
        BallQuery<DTYPE_XYZ> op;
        op.Init(center_xyz, xyz, idx, tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num, tiling_data.batch_size, tiling_data.num_points, tiling_data.num_centers);
        op.Process();
    }
    else if (tiling_data.type == 0) {
        BruteForce<DTYPE_XYZ> op;
        op.Init(center_xyz, xyz, idx, tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num, tiling_data.batch_size, tiling_data.num_points, tiling_data.num_centers);
        op.Process();
    }
    else if (tiling_data.type == 1) {
        BruteForceStack<DTYPE_XYZ> op;
        op.Init(center_xyz, xyz, center_xyz_batch, xyz_batch_cnt, idx, tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num, tiling_data.batch_size, tiling_data.num_points, tiling_data.num_centers);
        op.Process();
    }
}