#include "kernel_operator.h"
using namespace AscendC;
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
        if (this->R > batch_size) {
            this->R = batch_size;
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
        for (int index = total_size / batch_size * L, i = L; i < R; ++i) {
            for (int j = 0; j < num_groups; ++j) {
                for (int k = 0; k < length; ++k) {
                    T val = Gm_x.GetValue(index++);
                    mean[(i - L) * num_groups + j] += (float)val / length;
                }
            }
        }
        for (int index = total_size / batch_size * L, i = L; i < R; ++i) {
            for (int j = 0; j < num_groups; ++j) {
                float avg = mean[(i - L) * num_groups + j];
                for (int x = 0; x < chunk_size; ++x) {
                    float sum = 0;
                    for (int k = 0; k < length / chunk_size; ++k) {
                        float val = Gm_x.GetValue(index++);
                        sum += (val - avg) * (val - avg);
                    }
                    rstd[(i - L) * num_groups + j] += sum / length;
                }
            }
        }
        auto block_size = num_channels / num_groups;
        for (int index = total_size / batch_size * L, i = L; i < R; ++i) {
            for (int j = 0; j < num_channels; ++j) {
                float avg = mean[(i - L) * num_groups + j / block_size];
                float var = rstd[(i - L) * num_groups + j / block_size];
                float gm = Gm_gamma.GetValue(j), bt = Gm_beta.GetValue(j);
                for (int k = 0; k < total_size / batch_size / num_channels; ++k) {
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
    BruteForce<DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, tiling_data.span, tiling_data.chunk_size, tiling_data.batch_size, tiling_data.num_groups, tiling_data.num_channels, tiling_data.total_size, tiling_data.epsilon);
    op.Process();
}
