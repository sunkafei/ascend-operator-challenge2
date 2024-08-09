#include "kernel_operator.h"
using namespace AscendC;
constexpr int MAX_GROUP = 64;
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, int32_t batch_size, int32_t num_groups, int32_t num_channels, int32_t total_size, float epsilon) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->batch_size = batch_size;
        this->num_groups = num_groups;
        this->num_channels = num_channels;
        this->total_size = total_size;
        this->epsilon = epsilon;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, total_size);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, num_channels);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, num_channels);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, total_size);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, num_groups);
        Gm_rstd.SetGlobalBuffer((__gm__ T*)rstd, num_groups);
    }
    __aicore__ inline void Process() {
        /*float mean[MAX_GROUP] = {};
        for (int index = 0, i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_groups; ++j) {
                for (int k = 0; k < total_size / batch_size / num_groups; ++k) {
                    T val = Gm_x.GetValue(index++);
                    mean[j] += (float)val;
                }
            }
        }
        for (int j = 0; j < num_groups; ++j) {
            mean[j] /= total_size / num_groups;
            Gm_mean.SetValue(j, (T)mean[j]);
        }
        float rstd[MAX_GROUP] = {};
        for (int index = 0, i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_groups; ++j) {
                float avg = mean[j];
                for (int k = 0; k < total_size / batch_size / num_groups; ++k) {
                    float val = Gm_x.GetValue(index++);
                    rstd[j] += (val - avg) * (val - avg);
                }
            }
        }
        for (int j = 0; j < num_groups; ++j) {
            rstd[j] /= total_size / num_groups;
            Gm_rstd.SetValue(j, (T)rstd[j]);
        }
        auto block_size = num_channels / num_groups;
        for (int index = 0, i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_channels; ++j) {
                float avg = mean[j / block_size], var = rstd[j / block_size];
                float gm = Gm_gamma.GetValue(j), bt = Gm_beta.GetValue(j);
                for (int k = 0; k < total_size / batch_size / num_channels; ++k) {
                    float x = Gm_x.GetValue(index);
                    float result = gm * ((x - avg) / sqrt(var + epsilon)) + bt;
                    Gm_y.SetValue(index++, (T)result);
                }
            }
        }*/
        Gm_y.SetValue(0, (T)0.0f);
        Gm_y.SetValue(1, (T)0.0f);
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE>(Gm_y);
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_rstd;
    int32_t batch_size, num_groups, num_channels, total_size;
    float epsilon;
};
extern "C" __global__ __aicore__ void group_norm_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    BruteForce<DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, tiling_data.batch_size, tiling_data.num_groups, tiling_data.num_channels, tiling_data.total_size, tiling_data.epsilon);
    op.Process();
}