#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 4;                                     // tensor num for each queue

template<typename T, unsigned opType> class KernelDeepToSpace {
public:
    __aicore__ inline KernelDeepToSpace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape, uint32_t* bit=nullptr)
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
        this->bit = bit;

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
        }else if constexpr (opType == 4){
            auto mod2 = this->shape[2] - 1;
            auto mod3 = this->bs - 1;
            auto mod4 = this->shape[3] / this->bs - 1;
            auto C = this->bit[3] - this->bit[4] - this->bit[4];
            auto div2 = this->bit[2] + this->bit[3];
            auto div3 = this->bit[2] + this->bit[4] + C;
            auto div4 = this->bit[3] - this->bit[4];
            auto mul3 = this->bit[3];

            div2 = ~((1 << div2) - 1) ^ mod4;
            mod2 <<= div4;
            mul3 = mul3 - div4;
            mod3 <<= div3;
            auto mul4 = div3 - div4;

            for(uint32_t i=st;i<ed;i++){
                auto w = (i & mod2) << mul3;
                auto x = (i & mod3) >> mul4;
                auto hy = i & div2;

                zLocal.SetValue(i - st, xGm.GetValue(hy ^ w ^ x));
            }
        }else if constexpr (opType == 5){
            auto C = this->shape[3] / this->bs / this->bs;
            auto div2 = this->shape[2] * this->shape[3];
            auto div3 = this->shape[2] * this->bs * C;
            auto div4 = this->bs * C;
            auto mod2 = this->shape[2] - 1;

            auto mul3 = this->shape[3];
            for(uint32_t i=st;i<ed;i++){
                auto w = (i / div4) & mod2;

                auto h = i / div2 * div2;
                auto x = (i - h) / div3;
                auto y = i % div4;

                zLocal.SetValue(i - st, xGm.GetValue(h + w * mul3 + x * div4 + y));
            }
        }else if constexpr (opType == 6){
            auto mod4 = this->shape[3] / this->bs - 1;
            auto C = this->shape[3] / this->bs / this->bs;
            auto div2 = this->shape[2] * this->shape[3];
            auto div3 = this->shape[2] * this->bs * C;
            auto div4 = this->bit[3] - this->bit[4];

            auto mul3 = this->bit[3];
            for(uint32_t i=st;i<ed;i++){
                auto w = (i >> div4) % this->shape[2];

                auto h = i / div2 * div2;
                auto x = (i - h) / div3;
                auto y = i & mod4;

                zLocal.SetValue(i - st, xGm.GetValue(h + w * mul3 + (x << div4) + y));
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
    uint32_t* bit;
    uint32_t st, ed;
    LocalTensor<T> signbit;
};

// template<typename T, unsigned opType> class KernelDeepToSpace2 {
// public:
//     __aicore__ inline KernelDeepToSpace2() {}
//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape, uint32_t batch)
//     {
//         ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
//         this->totalLength = totalLength;
//         this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
//         this->batch = batch;

//         this->st = core_size * GetBlockIdx();
//         this->ed = this->st + this->blockLength;

//         this->tileLength = block_size;

//         this->bs = bs;
//         this->shape = shape;

//         auto startPointer = core_size * GetBlockIdx() * block_size;
//         this->startPointer = startPointer;

//         // get start index for current core, core parallel
//         xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, block_size * core_size);
//         zGm.SetGlobalBuffer((__gm__ T*)z, totalLength);

//         this->tileNum = core_size / batch;

//         this->transposeParams1.nSize = 1;
//         this->transposeParams1.cSize = batch;
//         this->transposeParams1.hSize = 1;
//         this->transposeParams1.wSize = shape[3];
//         this->transposeParams1.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;

//         this->transposeParams2.nSize = this->bs;
//         this->transposeParams2.cSize = shape[3] / this->bs;
//         this->transposeParams2.hSize = 1;
//         this->transposeParams2.wSize = batch;
//         this->transposeParams2.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;

//         // pipe alloc memory to queue, the unit is Bytes
//         pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
//         pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
//         pipe.InitBuffer(tmpQueue, BUFFER_NUM, this->tileLength * sizeof(T));
//     }
//     __aicore__ inline void Process()
//     {
        
//         // loop count need to be doubled, due to double buffer
//         int32_t loopCount = this->tileNum;
//         // tiling strategy, pipeline parallel
//         for (int32_t i = 0; i < loopCount-1; i++) {
//             CopyIn(i, this->tileLength);
//             Compute(i, this->tileLength);
//             CopyOut(i, this->tileLength);
//         }
//         auto length = this->ed - this->st - this->tileLength * (loopCount - 1);
//         CopyIn(loopCount - 1, length);
//         Compute(loopCount - 1, length);
//         CopyOut(loopCount - 1, length);
//     }

// private:
//     __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
//     {
//         // alloc tensor from queue memory
//         LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
//         // copy progress_th tile from global tensor to local tensor
//         DataCopy(xLocal, xGm[progress * this->tileLength], length);
//         // enque input tensors to VECIN queue
//         inQueueX.EnQue(xLocal);
//     }
//     __aicore__ inline void Compute(int32_t progress, uint32_t length)
//     {
//         // deque input tensors from VECIN queue
//         LocalTensor<T> xLocal = inQueueX.DeQue<T>();
//         LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
//         LocalTensor<uint8_t> stackBuffer = tmpQueue.AllocTensor<uint8_t>();

//         Transpose(xLocal, xLocal, stackBuffer, this->transposeParams1);
//         Transpose(zLocal, xLocal, stackBuffer, this->transposeParams2);

//         // enque the output tensor to VECOUT queue
//         outQueueZ.EnQue<T>(zLocal);
//         // free input tensors for reuse
//         inQueueX.FreeTensor(xLocal);
//     }
//     __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
//     {
//         // deque output tensor from VECOUT queue
//         LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
//         // copy progress_th tile from local tensor to global tensor
//         length /= this->bs;

//         auto bh = (this->st + progress) * this->batch / this->shape[2];
//         auto w = (this->st + progress) * this->batch % this->shape[2];
//         auto pointer = bh * this->shape[2] * this->shape[3] + w * shape[3] / this->bs;

//         for(int i=0;i<this->bs;i++){
//             DataCopy(zGm[pointer + i * this->shape[2] * this->shape[3] / this->bs], zLocal[i * length], length);
//         }
//         // free output tensor for reuse
//         outQueueZ.FreeTensor(zLocal);
//     }

// private:
//     TPipe pipe;
//     // create queues for input, in this case depth is equal to buffer num
//     TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
//     // create queue for output, in this case depth is equal to buffer num
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
//     TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue;
//     GlobalTensor<T> xGm;
//     GlobalTensor<T> zGm;
//     uint32_t blockLength;
//     uint32_t tileNum;
//     uint32_t tileLength;
//     uint32_t totalLength;
//     uint32_t startPointer;
//     uint32_t bs;
//     uint32_t* shape;
//     uint32_t batch;
//     uint32_t st, ed;
//     TransposeParamsExt transposeParams1, transposeParams2;
//     LocalTensor<T> signbit;
// };

template<typename T, unsigned opType> class KernelDeepToSpace2 {
public:
    __aicore__ inline KernelDeepToSpace2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape, uint32_t* bit)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->bit = bit;
        this->core_size = core_size;

        this->st = core_size * GetBlockIdx();
        this->ed = this->st + this->blockLength;

        this->tileLength = block_size;

        this->bs = bs;
        this->shape = shape;

        auto startPointer = core_size * GetBlockIdx() * block_size;
        this->startPointer = startPointer;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, block_size * this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalLength);

        this->tileNum = this->blockLength;

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    // __aicore__ inline void Process()
    // {
    //     // loop count need to be doubled, due to double buffer
    //     int32_t loopCount = this->tileNum;
    //     auto length = this->tileLength;
    //     // tiling strategy, pipeline parallel
    //     for (int32_t i = 0; i < loopCount; i++) {
    //         auto progress = i;
    //         {
    //             LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    //             DataCopy(xLocal, xGm[progress * this->tileLength], length);
    //             event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    //             SetFlag<HardEvent::MTE2_MTE3>(eventIDSToMTE3);
    //             WaitFlag<HardEvent::MTE2_MTE3>(eventIDSToMTE3);
    //             inQueueX.EnQue(xLocal);
    //         }
    //         {
    //             LocalTensor<T> xLocal = inQueueX.DeQue<T>();

    //             auto i = this->st + progress;
    //             auto hy = i / (this->shape[2] * this->shape[3] / length) * (this->shape[2] * this->shape[3] / length);
    //             auto x = i % this->bs * this->shape[2];
    //             auto w = i / this->bs % this->shape[2];

    //             DataCopy(zGm[(hy ^ x ^ w) * this->tileLength], xLocal, length);
    //             // DataCopy(zGm[j * this->tileLength], xLocal, length);
    //             // free output tensor for reuse
    //             inQueueX.FreeTensor(xLocal);
    //         }
    //     }
    // }
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            // Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDSToMTE3);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    // __aicore__ inline void Compute(int32_t progress, uint32_t length)
    // {
    //     // deque input tensors from VECIN queue
    //     LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    //     LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

    //     DataCopy(zLocal, xLocal, length);

    //     // enque the output tensor to VECOUT queue
    //     outQueueZ.EnQue<T>(zLocal);
    //     // free input tensors for reuse
    //     inQueueX.FreeTensor(xLocal);
    // }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        // LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        LocalTensor<T> zLocal = inQueueX.DeQue<T>();
        // copy progress_th tile from local tensor to global tensor
        // auto mod1 = this->bs - 1;
        // auto mod2 = this->shape[2] - 1;
        // auto div2 = this->bit[4];
        // auto div3 = ~(1 << (this->bit[4] + this->bit[2])) - 1;
        // auto i = this->st + progress;
        // auto hy = i & div3;
        // auto x = i & mod1;
        // auto w = (i >> div2) & mod2;

        auto i = this->st + progress;
        auto hy = i / (this->shape[2] * this->shape[3] / length) * (this->shape[2] * this->shape[3] / length);
        auto x = i % this->bs * this->shape[2];
        auto w = i / this->bs % this->shape[2];
        SyncAll();

        DataCopy(zGm[(hy ^ x ^ w) * this->tileLength], zLocal, length);
        // free output tensor for reuse
        // outQueueZ.FreeTensor(zLocal);
        inQueueX.FreeTensor(zLocal);
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
    uint32_t core_size;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t startPointer;
    uint32_t bs;
    uint32_t* shape;
    uint32_t* bit;
    uint32_t batch;
    uint32_t st, ed;
    TransposeParamsExt transposeParams1, transposeParams2;
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
    }else if(tiling_data.type == 4){
        KernelDeepToSpace <DTYPE_X, 4> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.bit);
        op.Process();
    }else if(tiling_data.type == 5){
        KernelDeepToSpace <DTYPE_X, 5> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.bit);
        op.Process();
    }else if(tiling_data.type == 6){
        KernelDeepToSpace <DTYPE_X, 6> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.bit);
        op.Process();
    }else if(tiling_data.type == 7){
        KernelDeepToSpace2 <DTYPE_X, 7> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.bit);
        op.Process();
    }
}