#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename T> __aicore__ inline void Reduce(const LocalTensor<T> &x, uint32_t length) {
    while (length > 32 / sizeof(T)) {
        length >>= 1;
        Add(x, x, x[length], length);
        PipeBarrier<PIPE_V>();
    }
    BlockReduceSum(x, x, 1, 32 / sizeof(T), 1, 1, 8);
}

template<typename T> class KernelPdist {
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, float p, uint32_t* shape)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() / 2 == GetBlockIdx() / 2 + 1 ? core_remain : 0);
        this->core_size = core_size;
        this->p = p;
        this->invp = 1.0f / p;

        this->st = core_size * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        this->ed = this->st + core_size + (GetBlockIdx() < core_remain ? 1 : 0);

        this->tileLength = block_size;

        this->ALIGN_NUM = ALIGN_NUM;
        this->shape = shape;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalLength);

        this->tileNum = this->blockLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->shape[1] * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->shape[1] * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->ALIGN_NUM);
        pipe.InitBuffer(calcBuf, this->shape[1] * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int length = (this->shape[1] + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
        for(int i = st; i < ed; i++){
            for(int j = i + 1; j < this->shape[0]; j++){
                CopyIn(i, j, length);
                Compute(this->shape[1], length);
                CopyOut((2 * this->shape[0] - 2 - i + 1) * i / 2 + j - i - 1);
            }
        }
    }
    __aicore__ inline void CopyIn(int32_t i, int32_t j, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[i * this->shape[1]], length);
        DataCopy(yLocal, xGm[j * this->shape[1]], length);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t length, uint32_t length_aligned)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        Sub(xLocal, xLocal, yLocal, length);
        Abs(xLocal, xLocal, length);

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> tmp = calcBuf.Get<float>();
            Cast(tmp, xLocal, RoundMode::CAST_NONE, length);

            Ln(tmp, tmp, length);
            Muls(tmp, tmp, this->p, length);
            Exp(tmp, tmp, length);

            Sum(tmp, tmp, SumParams{1, length_aligned, length});
            Ln(tmp, tmp, this->ALIGN_NUM);
            Muls(tmp, tmp, this->invp, this->ALIGN_NUM);
            Exp(tmp, tmp, this->ALIGN_NUM);

            Cast(zLocal, tmp, RoundMode::CAST_NONE, this->ALIGN_NUM);
        }else{
            Ln(xLocal, xLocal, length);
            Muls(xLocal, xLocal, this->p, length);
            Exp(xLocal, xLocal, length);

            Sum(zLocal, xLocal, SumParams{1, length_aligned, length});
            Ln(zLocal, zLocal, this->ALIGN_NUM);
            Muls(zLocal, zLocal, this->invp, this->ALIGN_NUM);
            Exp(zLocal, zLocal, this->ALIGN_NUM);
        }


        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();

        T x = progress;
        // zGm.SetValue(progress, x);
        zGm.SetValue(progress, zLocal.GetValue(0));
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(zGm);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<TPosition::VECCALC> calcBuf;
    // TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t core_size;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t startPointer;
    uint32_t* shape;
    uint32_t st, ed;
    float p, invp;
    uint32_t ALIGN_NUM;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    
    for(int i = 0; i < 2; i++){
        KernelPdist <DTYPE_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.p, tiling_data.shape);
        op.Process();
    }
}