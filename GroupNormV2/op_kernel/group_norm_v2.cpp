#include "kernel_operator.h"
using namespace AscendC;
template<typename T> __aicore__ inline void Reduce(const LocalTensor<T> &y, const LocalTensor<T> &x, uint32_t group_size) {
    Sum(y, x, SumParams{1, group_size, group_size});
}
template<typename T> class GroupNormV2Kernal {
public:
    __aicore__ inline GroupNormV2Kernal() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, int32_t tile_length, int32_t span, int32_t chunk_size, int32_t batch_size, int32_t num_groups, int32_t num_channels, int32_t total_size, float epsilon) {
        auto num_cores = GetBlockNum();
        auto block_index = GetBlockIdx();
        this->chunk_size = chunk_size;
        this->batch_size = batch_size;
        this->num_groups = num_groups;
        this->num_channels = num_channels;
        this->total_size = total_size;
        this->epsilon = epsilon;
        this->L = block_index * span;
        this->R = (block_index + 1) * span;
        if (this->R > total_size / tile_length) {
            this->R = total_size / tile_length;
        }
        this->block_size = num_channels / num_groups;
        this->tile_length = tile_length;
        this->data_copy_size = (sizeof(T) * batch_size * num_groups + 31) & ~31;
        this->group_size = total_size / batch_size / num_groups;
        this->channel_size = total_size / batch_size / num_channels;
        this->group_tiles = group_size / tile_length;

        auto range = (batch_size * num_groups - 1) / num_cores + 1;
        this->L2 = block_index * range;
        this->R2 = (block_index + 1) * range;
        if (this->R2 > batch_size * num_groups) {
            this->R2 = batch_size * num_groups;
        }

        Gm_x.SetGlobalBuffer((__gm__ T*)x, total_size);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, num_channels);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, num_channels);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, total_size);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, batch_size * num_groups);
        Gm_rstd.SetGlobalBuffer((__gm__ T*)rstd, batch_size * num_groups);
        pipe.InitBuffer(Q_mean, 1, data_copy_size);
        pipe.InitBuffer(Q_rstd, 1, data_copy_size);
        pipe.InitBuffer(B_sumv, data_copy_size);
        pipe.InitBufPool(stage1, 4 * sizeof(T) * tile_length);
        pipe.InitBufPool(stage2, 4 * sizeof(T) * tile_length, stage1);
    }
    __aicore__ inline void Preprocess() {
        TQue<QuePosition::VECIN, 2> Q_x;
        stage1.InitBuffer(Q_x, 2, 2 * sizeof(T) * tile_length);
        auto sumv = B_sumv.Get<T>();
        auto mean = Q_mean.AllocTensor<T>();
        auto rstd = Q_rstd.AllocTensor<T>();
        Duplicate(mean, T(0), data_copy_size);
        Duplicate(rstd, T(0), data_copy_size);
        float sum = 0, sum2 = 0;
        int last = 0;
        for (int32_t i = L; i < R; ) {
            auto index = i / group_tiles;
            auto offset = i * tile_length;
            int32_t block_length;
            if (i != R - 1 && i % group_tiles != group_tiles - 1) [[likely]] {
                block_length = tile_length * 2;
                i += 2;
            }
            else {
                block_length = tile_length;
                i += 1;
            }
            if (index != last) [[unlikely]] {
                mean.SetValue(last, sum / group_size);
                rstd.SetValue(last, sum2 / group_size);
                last = index;
                sum = sum2 = 0;
            }
            {
                LocalTensor<T> x = Q_x.AllocTensor<T>();
                DataCopy(x, Gm_x[offset], block_length);
                Q_x.EnQue(x);
            }
            {
                LocalTensor<T> x = Q_x.DeQue<T>();
                Reduce(sumv, x, block_length);
                sum += (float)sumv.GetValue(0);
                Mul(x, x, x, block_length);
                Reduce(sumv, x, block_length);
                sum2 += (float)sumv.GetValue(0);
                Q_x.FreeTensor(x);
            }
        }
        mean.SetValue(last, sum / group_size);
        rstd.SetValue(last, sum2 / group_size);
        SetAtomicAdd<T>();
        DataCopy(Gm_mean, mean, data_copy_size);
        DataCopy(Gm_rstd, rstd, data_copy_size);
        Q_mean.FreeTensor(mean);
        Q_rstd.FreeTensor(rstd);
        stage1.Reset();
    }
    __aicore__ inline void Process() { // 872
        Preprocess();
        TQue<QuePosition::VECIN, 2> Q_x;
        TQue<QuePosition::VECOUT, 2> Q_y;
        stage2.InitBuffer(Q_x, 2, sizeof(T) * tile_length);
        stage2.InitBuffer(Q_y, 2, sizeof(T) * tile_length);
        // mean = Q_mean.AllocTensor<T>();
        // rstd = Q_rstd.AllocTensor<T>();
        // Duplicate(rstd, T(0), data_copy_size);
        // SyncAll();
        // for (int i = L2; i < R2; ++i) {
        //     float avg = Gm_mean.GetValue(i);
        //     rstd.SetValue(i, -avg * avg);
        // }
        // DataCopy(Gm_rstd, rstd, data_copy_size);
        // Q_mean.FreeTensor(mean);
        // Q_rstd.FreeTensor(rstd);

        auto mean = Q_mean.AllocTensor<T>();
        auto rstd = Q_rstd.AllocTensor<T>();
        SyncAll();
        SetAtomicNone();
        float avg, var, gm, bt;
        int last = -1, last2 = -1;
        for (int i = L; i < R; ++i) {
            auto index = i / group_tiles;
            auto index2 = i * tile_length / channel_size % num_channels;
            if (index2 != last2) [[unlikely]] {
                last2 = index2;
                gm = Gm_gamma.GetValue(index2);
                bt = Gm_beta.GetValue(index2);
                if (index != last) [[unlikely]] {
                    last = index;
                    avg = Gm_mean.GetValue(index);
                    var = sqrt((float)Gm_rstd.GetValue(index) + epsilon);
                }
                gm = gm / var;
                bt += gm * -avg;
            }
            {
                LocalTensor<T> x = Q_x.AllocTensor<T>();
                DataCopy(x, Gm_x[i * tile_length], tile_length);
                Q_x.EnQue(x);
            }
            {
                LocalTensor<T> y = Q_y.AllocTensor<T>();
                LocalTensor<T> x = Q_x.DeQue<T>();
                Muls(x, x, T(gm), tile_length);
                Adds(y, x, T(bt), tile_length);
                Q_y.EnQue<T>(y);
                Q_x.FreeTensor(x);
            }
            {
                LocalTensor<T> y = Q_y.DeQue<T>();
                DataCopy(Gm_y[i * tile_length], y, tile_length);
                Q_y.FreeTensor(y);
            }
        }
        Q_mean.FreeTensor(mean);
        Q_rstd.FreeTensor(rstd);
        stage2.Reset();
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_rstd;
    int32_t L, R, block_size, chunk_size, batch_size, num_groups, num_channels, total_size;
    int32_t tile_length, data_copy_size, group_size, channel_size;
    int32_t group_tiles, L2, R2;
    float epsilon;
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> Q_mean, Q_rstd;
    TBuf<QuePosition::VECCALC> B_sumv;
    TBufPool<TPosition::VECCALC> stage1, stage2;
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