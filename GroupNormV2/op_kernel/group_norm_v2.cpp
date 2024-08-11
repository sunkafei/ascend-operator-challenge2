#include "kernel_operator.h"
using namespace AscendC;
template<typename T> __aicore__ inline void Reduce(const LocalTensor<T> &x, int32_t group_size) {
    if (group_size * sizeof(T) > 256) {
        const int number = 256 / sizeof(T);
        group_size /= number;
        WholeReduceSum(x, x, number, group_size, 1, 1, 8);
    }
    WholeReduceSum(x, x, group_size, 1, 1, 1, 8);
}
template<typename T> class GroupNormV2Kernal {
public:
    __aicore__ inline GroupNormV2Kernal() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, int32_t tile_length, int32_t span, int32_t chunk_size, int32_t batch_size, int32_t num_groups, int32_t num_channels, int32_t total_size, float epsilon) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->chunk_size = chunk_size;
        this->batch_size = batch_size;
        this->num_groups = num_groups;
        this->num_channels = num_channels;
        this->total_size = total_size;
        this->epsilon = epsilon;
        this->L = GetBlockIdx() * span;
        this->R = (GetBlockIdx() + 1) * span;
        if (this->R > batch_size * num_groups) {
            this->R = batch_size * num_groups;
        }
        if (L >= R) {
            return;
        }
        this->block_size = num_channels / num_groups;
        this->length = total_size / batch_size / num_groups;
        this->tile_length = tile_length;
        this->splits = length / tile_length;

        Gm_x.SetGlobalBuffer((__gm__ T*)x, total_size);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, num_channels);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, num_channels);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, total_size);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, num_groups);
        Gm_rstd.SetGlobalBuffer((__gm__ T*)rstd, num_groups);
        pipe.InitBuffer(Q_x, 4, sizeof(T) * tile_length);
        pipe.InitBuffer(Q_y, 4, sizeof(T) * tile_length);
    }
    __aicore__ inline void Process() {
        if (L >= R) {
            return;
        }
        float mean[512] = {};
        float rstd[512] = {};
        for (int i = L; i < R; ++i) {
            for (int j = 0; j < splits; ++j) {
                {
                    LocalTensor<T> x = Q_x.AllocTensor<T>();
                    DataCopy(x, Gm_x[i * length + j * tile_length], tile_length);
                    Q_x.EnQue(x);
                }
                {
                    LocalTensor<T> x = Q_x.DeQue<T>();
                    Reduce(x, tile_length);
                    mean[i] += (float)x.GetValue(0) / length;
                    Q_x.FreeTensor(x);
                }
            }
        }
        for (int index = L * length, i = L; i < R; ++i) {
            float avg = mean[i];
            for (int j = 0; j < splits; ++j) {
                {
                    LocalTensor<T> x = Q_x.AllocTensor<T>();
                    DataCopy(x, Gm_x[i * length + j * tile_length], tile_length);
                    Q_x.EnQue(x);
                }
                {
                    LocalTensor<T> x = Q_x.DeQue<T>();
                    Adds(x, x, T(-avg), tile_length);
                    Mul(x, x, x, tile_length);
                    Reduce(x, tile_length);
                    rstd[i] += (float)x.GetValue(0) / length;
                    Q_x.FreeTensor(x);
                }
            }
        }
        for (int i = L; i < R; ++i) {
            float avg = mean[i];
            float var = sqrt(rstd[i] + epsilon);
            for (int j = 0; j < block_size; ++j) {
                float gm = Gm_gamma.GetValue(i % num_groups * block_size + j);
                float bt = Gm_beta.GetValue(i % num_groups * block_size + j);
                float coef = gm / var;
                for (int k = 0; k < splits / block_size; ++k) {
                    auto index = i * length + (j * splits / block_size + k) * tile_length;
                    {
                        LocalTensor<T> x = Q_x.AllocTensor<T>();
                        DataCopy(x, Gm_x[index], tile_length);
                        Q_x.EnQue(x);
                    }
                    {
                        LocalTensor<T> y = Q_y.AllocTensor<T>();
                        LocalTensor<T> x = Q_x.DeQue<T>();
                        Adds(x, x, T(-avg), tile_length);
                        Muls(x, x, T(coef), tile_length);
                        Adds(y, x, T(bt), tile_length);
                        Q_y.EnQue<T>(y);
                        Q_x.FreeTensor(x);
                    }
                    {
                        LocalTensor<T> y = Q_y.DeQue<T>();
                        DataCopy(Gm_y[index], y, tile_length);
                        Q_y.FreeTensor(y);
                    }
                }
            }
        }
        //DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(Gm_y);
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_rstd;
    int32_t L, R, block_size, chunk_size, batch_size, num_groups, num_channels, total_size;
    int32_t length, tile_length, splits;
    float epsilon;
    TPipe pipe;
    TQue<QuePosition::VECIN, 4> Q_x;
    TQue<QuePosition::VECOUT, 4> Q_y;
};
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, int32_t span, int32_t chunk_size, int32_t batch_size, int32_t num_groups, int32_t num_channels, int32_t total_size, float epsilon) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->chunk_size = chunk_size;
        this->batch_size = batch_size;
        this->num_groups = num_groups;
        this->num_channels = num_channels;
        this->total_size = total_size;
        this->epsilon = epsilon;
        this->L = GetBlockIdx() * span;
        this->R = (GetBlockIdx() + 1) * span;
        if (this->R > batch_size * num_groups) {
            this->R = batch_size * num_groups;
        }
        Gm_x.SetGlobalBuffer((__gm__ T*)x, total_size);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, num_channels);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, num_channels);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, total_size);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, num_groups);
        Gm_rstd.SetGlobalBuffer((__gm__ T*)rstd, num_groups);
    }
    __aicore__ inline void Process() {
        if (L >= R) {
            return;
        }
        float mean[512] = {};
        float rstd[512] = {};
        const int length = total_size / batch_size / num_groups;
        for (int index = L * length, i = L; i < R; ++i) {
            for (int j = 0; j < length; ++j) {
                T val = Gm_x.GetValue(index++);
                mean[i] += (float)val / length;
            }
        }
        for (int index = L * length, i = L; i < R; ++i) {
            float avg = mean[i];
            for (int x = 0; x < chunk_size; ++x) {
                float sum = 0;
                for (int y = 0; y < length / chunk_size; ++y) {
                    float val = Gm_x.GetValue(index++);
                    sum += (val - avg) * (val - avg);
                }
                rstd[i] += sum / length;
            }
        }
        const auto block_size = num_channels / num_groups;
        for (int index = L * length, i = L; i < R; ++i) {
            float avg = mean[i];
            float var = rstd[i];
            for (int j = 0; j < block_size; ++j) {
                float gm = Gm_gamma.GetValue(i % num_groups * block_size + j);
                float bt = Gm_beta.GetValue(i % num_groups * block_size + j);
                for (int k = 0; k < length / block_size; ++k) {
                    float x = Gm_x.GetValue(index);
                    float result = gm * ((x - avg) / sqrt(var + epsilon)) + bt;
                    Gm_y.SetValue(index++, (T)result);
                }
            }
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(Gm_y);
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_rstd;
    int32_t L, R, chunk_size, batch_size, num_groups, num_channels, total_size;
    float epsilon;
};
extern "C" __global__ __aicore__ void group_norm_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.tile_length == -1) {
        BruteForce<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, tiling_data.span, tiling_data.chunk_size, tiling_data.batch_size, tiling_data.num_groups, tiling_data.num_channels, tiling_data.total_size, tiling_data.epsilon);
        op.Process();
    }
    else {
        GroupNormV2Kernal<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, rstd, tiling_data.tile_length, tiling_data.span, tiling_data.chunk_size, tiling_data.batch_size, tiling_data.num_groups, tiling_data.num_channels, tiling_data.total_size, tiling_data.epsilon);
        op.Process();
    }
}
