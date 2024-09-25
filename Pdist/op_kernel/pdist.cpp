#include "kernel_operator.h"
using namespace AscendC;
template<typename T> __aicore__ inline void GroupReduce(const LocalTensor<T> &y, const LocalTensor<T> &x, int32_t group_size, int32_t group_count) {
    static constexpr int32_t SIZE = sizeof(T);
    static constexpr int32_t ALIGN = 32 / SIZE;
    const int32_t factor = group_size / (group_size & -group_size);
    int32_t number = (256 / SIZE) / factor;
    number |= (number >> 1);
    number |= (number >> 2);
    number |= (number >> 4);
    int32_t reduceCount = (number ^ (number >> 1)) * factor;
    if (group_size / reduceCount > 1) {
        if (group_size / reduceCount % ALIGN) {
            reduceCount /= ALIGN * reduceCount / group_size;
        }
        int32_t repeatTimes = group_count * group_size / reduceCount;
        int32_t repStride = (reduceCount * SIZE - 1) / 32 + 1;
        WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
        group_size /= reduceCount;
    }
    WholeReduceSum(y, x, group_size, group_count, 1, 1, group_size * SIZE / 32);
}
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, int n, int m) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->p = p;
        this->n = n;
        this->m = m;
        this->copym = (m * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
        xGm.SetGlobalBuffer((__gm__ T*)x, n * m);
        yGm.SetGlobalBuffer((__gm__ T*)y, n * (n - 1) / 2);
        pipe.InitBuffer(QA, 2, 1024 * 4);
        pipe.InitBuffer(QB, 2, 1024 * 4);
        pipe.InitBuffer(QY, 2, 1024 * 4);
    }
    __aicore__ inline void Process() {
        int index = 0;
        for (int i = 0; i < n; ++i) {
            LocalTensor<T> a_tmp = QA.AllocTensor<T>();
            DataCopy(a_tmp, xGm[i * m], copym);
            QA.EnQue(a_tmp);
            LocalTensor<T> a = QA.DeQue<T>();
            LocalTensor<T> y = QY.AllocTensor<T>();
            for (int j = i + 1; j < n; ++j) {
                LocalTensor<T> b_tmp = QB.AllocTensor<T>();
                DataCopy(b_tmp, xGm[j * m], copym);
                QB.EnQue(b_tmp);
                LocalTensor<T> b = QB.DeQue<T>();
                Sub(b, a, b, copym);
                Abs(b, b, copym);
                Ln(b, b, copym);
                Muls(b, b, (T)p, copym);
                Exp(b, b, copym);
                Sum(y[j - (i + 1)], b, SumParams{1, copym, m});
                QB.FreeTensor(b);
            }
            int length = n - i - 1;
            length = (length * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
            Ln(y, y, length);
            Muls(y, y, (T)(1.0f / p), length);
            Exp(y, y, length);
            QY.EnQue<T>(y);
            QA.FreeTensor(a);
            LocalTensor<T> y_tmp = QY.DeQue<T>();

            DataCopy(yGm[index], y_tmp, length);
            QY.FreeTensor(y_tmp);
            index += n - i - 1;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> QA, QB;
    TQue<QuePosition::VECOUT, 2> QY;
    GlobalTensor<T> xGm, yGm;
    float p;
    uint32_t n;
    uint32_t m, copym;
};
template<> class BruteForce<float16_t> {
public:
    using T = float16_t;
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, int n, int m) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->p = p;
        this->n = n;
        this->m = m;
        this->copym = (m * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
        xGm.SetGlobalBuffer((__gm__ T*)x, n * m);
        yGm.SetGlobalBuffer((__gm__ T*)y, n * (n - 1) / 2);
        pipe.InitBuffer(QA, 2, 1024 * 4);
        pipe.InitBuffer(QB, 2, 1024 * 4);
        pipe.InitBuffer(QY, 2, 1024 * 4);
        pipe.InitBuffer(BA, 1024 * 4);
        pipe.InitBuffer(BB, 1024 * 4);
        pipe.InitBuffer(BY, 1024 * 4);
    }
    __aicore__ inline void Process() {
        auto a = BA.Get<float>(), b = BB.Get<float>(), y = BY.Get<float>();
        int index = 0;
        for (int i = 0; i < n; ++i) {
            LocalTensor<T> a_tmp = QA.AllocTensor<T>();
            DataCopy(a_tmp, xGm[i * m], copym);
            QA.EnQue(a_tmp);
            LocalTensor<T> a16 = QA.DeQue<T>();
            Cast(a, a16, RoundMode::CAST_NONE, copym);
            LocalTensor<T> y16 = QY.AllocTensor<T>();
            for (int j = i + 1; j < n; ++j) {
                LocalTensor<T> b_tmp = QB.AllocTensor<T>();
                DataCopy(b_tmp, xGm[j * m], copym);
                QB.EnQue(b_tmp);
                LocalTensor<T> b16 = QB.DeQue<T>();
                Cast(b, b16, RoundMode::CAST_NONE, copym);
                Sub(b, a, b, copym);
                Abs(b, b, copym);
                Ln(b, b, copym);
                Muls(b, b, p, copym);
                Exp(b, b, copym);
                Sum(y[j - (i + 1)], b, SumParams{1, copym, m});
                QB.FreeTensor(b16);
            }
            int length = n - i - 1;
            length = (length * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
            Ln(y, y, length);
            Muls(y, y, (1.0f / p), length);
            Exp(y, y, length);
            Cast(y16, y, RoundMode::CAST_NONE, copym);
            QY.EnQue<T>(y16);
            QA.FreeTensor(a16);
            LocalTensor<T> y_tmp = QY.DeQue<T>();

            DataCopy(yGm[index], y_tmp, length);
            QY.FreeTensor(y_tmp);
            index += n - i - 1;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> QA, QB;
    TQue<QuePosition::VECOUT, 2> QY;
    TBuf<QuePosition::VECCALC> BA, BB, BY;
    GlobalTensor<T> xGm, yGm;
    float p;
    uint32_t n;
    uint32_t m, copym;
};
class PdistKernal {
    using T = float;
    static constexpr int packNumber = 8;
public:
    __aicore__ inline PdistKernal() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, int n, int m, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->p = p;
        this->n = n;
        this->m = m;
        this->copym = (m * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
        this->st = core_size * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        this->ed = this->st + core_size + (GetBlockIdx() < core_remain ? 1 : 0);
        xGm.SetGlobalBuffer((__gm__ T*)x, n * m);
        yGm.SetGlobalBuffer((__gm__ T*)y, n * (n - 1) / 2);
        pipe.InitBuffer(QA, 2, 1024 * sizeof(T) * packNumber);
        pipe.InitBuffer(QB, 2, 1024 * sizeof(T) * packNumber);
        pipe.InitBuffer(QY, 2, 1024 * sizeof(T));
    }
    __aicore__ inline void Process() {
        for(int tt = 0; tt < 2; tt++){
            if(tt == 1){
                int x = st, y = ed;
                st = n - y;
                ed = n - x;
            }
            for (int i = st; i < ed; ++i) {
                int index = (2 * n - 2 - i + 1) * i / 2;
                LocalTensor<T> a_tmp = QA.AllocTensor<T>();
                DataCopy(a_tmp, xGm[i * m], copym);
                QA.EnQue(a_tmp);
                LocalTensor<T> a = QA.DeQue<T>();
                for (int i = 1; i < packNumber; i *= 2) {
                    Adds(a[i * m], a, T(0), copym * i);
                }
                LocalTensor<T> y = QY.AllocTensor<T>();
                int length = n - i - 1;
                length = (length * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
                Duplicate(y, 0.0f, length);
                PipeBarrier<PIPE_V>();
                for (int j = i + 1; j < n; j += packNumber) {
                    const int gsize = (packNumber < n - j ? packNumber : n - j);
                    LocalTensor<T> b_tmp = QB.AllocTensor<T>();
                    DataCopy(b_tmp, xGm[j * m], copym * gsize);
                    QB.EnQue(b_tmp);
                    LocalTensor<T> b = QB.DeQue<T>();
                    Sub(b, a, b, copym * gsize);
                    Mul(b, b, b, copym * gsize);
                    GroupReduce(y[j - (i + 1)], b, m, gsize);
                    QB.FreeTensor(b);
                }
                Sqrt(y, y, length);
                QY.EnQue<T>(y);
                QA.FreeTensor(a);
                LocalTensor<T> y_tmp = QY.DeQue<T>();

                SetAtomicAdd<T>();
                DataCopy(yGm[index], y_tmp, length);
                SetAtomicNone();
                QY.FreeTensor(y_tmp);
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> QA, QB;
    TQue<QuePosition::VECOUT, 2> QY;
    GlobalTensor<T> xGm, yGm;
    float p;
    uint32_t n;
    uint32_t m, copym;
    uint32_t st, ed;
};
extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (sizeof(DTYPE_X) == 4 && tiling_data.single_bits && tiling_data.p == 2.0f) {
        PdistKernal op;
        op.Init(x, y, tiling_data.p, tiling_data.n, tiling_data.m, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    else {
        BruteForce<DTYPE_X> op;
        op.Init(x, y, tiling_data.p, tiling_data.n, tiling_data.m);
        op.Process();
    }
}