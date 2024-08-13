#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 4;                                     // tensor num for each queue

template<typename T, unsigned opType> class KernelDeepToSpace {
public:
    __aicore__ inline KernelDeepToSpace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);

        this->st = core_size * GetBlockIdx();
        this->ed = this->st + this->blockLength;

        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        this->bs = bs;
        this->shape = shape;

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;
        this->startPointer = startPointer;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ T*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        // pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            // CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->ed - this->st - this->tileLength * (loopCount - 1);
        // CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    // __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    // {
    //     // alloc tensor from queue memory
    //     LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    //     // copy progress_th tile from global tensor to local tensor
    //     DataCopy(xLocal, xGm[progress * this->tileLength], length);
    //     // enque input tensors to VECIN queue
    //     inQueueX.EnQue(xLocal);
    // }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        auto st = this->st + progress * this->tileLength;
        auto ed = st + length;
        // deque input tensors from VECIN queue
        // LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        if constexpr (opType == 0){
            auto C = this->shape[1] / this->bs / this->bs;
            auto div1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto div2 = this->shape[2] * this->shape[3] * this->bs * this->bs;
            auto div3 = this->shape[3] * this->bs * this->bs;
            auto div4 = this->shape[3] * this->bs;
            auto div5 = this->bs;

            auto mul1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto mul2 = this->shape[2] * this->shape[3] * this->bs * C;
            auto mul3 = this->shape[2] * this->shape[3] * C;
            auto mul4 = this->shape[2] * this->shape[3];
            auto mul5 = this->shape[3];
            for(uint32_t i=st;i<ed;i++){
                // auto b = i / div1;
                // auto c = i / div2 % C;
                auto h = i / div3 % this->shape[2];
                auto x = i / div4 % this->bs;
                auto w = i / div5 % this->shape[3];
                auto y = i % this->bs;

                auto b = i / div1 * mul1;
                auto c = (i - b) / div2;

                zLocal.SetValue(i - st, xGm.GetValue(b + x * mul2 + y * mul3 + c * mul4 + h * mul5 + w));
                // zLocal.SetValue(i - st, xGm.GetValue(b * mul1 + x * mul2 + y * mul3 + c * mul4 + h * mul5 + w));
            }
        }else if constexpr (opType == 1){
            auto C = this->shape[1] / this->bs / this->bs;
            auto div1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto div2 = this->shape[2] * this->shape[3] * this->bs * this->bs;
            auto div3 = this->shape[3] * this->bs * this->bs;
            auto div4 = this->shape[3] * this->bs;
            auto div5 = this->bs;

            auto mul1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto mul2 = this->shape[2] * this->shape[3] * this->bs * this->bs;
            auto mul3 = this->shape[2] * this->shape[3] * this->bs;
            auto mul4 = this->shape[2] * this->shape[3];
            auto mul5 = this->shape[3];
            for(uint32_t i=st;i<ed;i++){
                // auto b = i / div1;
                // auto c = i / div2 % C;
                // auto h = i / div3 % this->shape[2];
                auto x = i / div4 % this->bs;
                auto w = i / div5 % this->shape[3];
                auto y = i % this->bs;

                
                auto c = i / div2 * mul2;
                auto h = (i - c) / div3;

                zLocal.SetValue(i - st, xGm.GetValue(c + x * mul3 + y * mul4 + h * mul5 + w));
                // zLocal.SetValue(i - st, xGm.GetValue(b * mul1 + c * mul2 + x * mul3 + y * mul4 + h * mul5 + w));
            }
        }else if constexpr (opType == 2){
            auto C = this->shape[3] / this->bs / this->bs;
            auto div1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto div2 = this->shape[2] * this->shape[3];
            auto div3 = this->shape[2] * this->bs * C;
            auto div4 = this->bs * C;
            auto div5 = C;

            auto mul1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto mul2 = this->shape[2] * this->shape[3];
            auto mul3 = this->shape[3];
            auto mul4 = C * this->bs;
            auto mul5 = C;
            for(uint32_t i=st;i<ed;i++){
                // auto b = i / div1;
                // auto h = i / div2 % this->shape[1];
                // auto x = i / div3 % this->bs;
                auto w = i / div4 % this->shape[2];
                // auto y = i / div5 % this->bs;
                // auto c = i % div5;

                auto h = i / div2 * mul2;
                auto x = (i - h) / div3;
                auto y = i % div4;

                zLocal.SetValue(i - st, xGm.GetValue(h + w * mul3 + x * mul4 + y));
                // zLocal.SetValue(i - st, xGm.GetValue(b * mul1 + h * mul2 + w * mul3 + x * mul4 + y * mul5 + c));
            }
        }else if constexpr (opType == 3){
            auto C = this->shape[3] / this->bs / this->bs;
            auto div1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto div2 = this->shape[2] * this->shape[3];
            auto div3 = this->shape[2] * this->bs * C;
            auto div4 = this->bs * C;
            auto div5 = C;

            auto mul1 = this->shape[1] * this->shape[2] * this->shape[3];
            auto mul2 = this->shape[2] * this->shape[3];
            auto mul3 = this->shape[3];
            auto mul4 = this->bs * this->bs;
            auto mul5 = this->bs;
            for(uint32_t i=st;i<ed;i++){
                // auto b = i / div1;
                // auto h = i / div2 % this->shape[1];
                // auto x = i / div3 % this->bs;
                auto w = i / div4 % this->shape[2];
                auto y = i / div5 % this->bs;
                auto c = i % div5;

                auto h = i / div2 * mul2;
                auto x = (i - h) / div3;

                zLocal.SetValue(i - st, xGm.GetValue(h + w * mul3 + c * mul4 + x * mul5 + y));
                // zLocal.SetValue(i - st, xGm.GetValue(b * mul1 + h * mul2 + w * mul3 + c * mul4 + x * mul5 + y));
            }
        }

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        // inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t startPointer;
    uint32_t bs;
    uint32_t* shape;
    uint32_t st, ed;
    LocalTensor<T> signbit;
};

extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(tiling_data.type == 0){
        KernelDeepToSpace <DTYPE_X, 0> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape);
        op.Process();
    }else if(tiling_data.type == 1){
        KernelDeepToSpace <DTYPE_X, 1> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape);
        op.Process();
    }else if(tiling_data.type == 2){
        KernelDeepToSpace <DTYPE_X, 2> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape);
        op.Process();
    }else if(tiling_data.type == 3){
        KernelDeepToSpace <DTYPE_X, 3> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape);
        op.Process();
    }
}