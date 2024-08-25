#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

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

template<typename T, unsigned opType> class KernelDeepToSpace2 {
public:
    __aicore__ inline KernelDeepToSpace2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape, uint32_t batch, uint32_t* bit=nullptr)
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
        this->ALIGN_NUM = ALIGN_NUM;
        this->shape = shape;
        this->batch = batch;

        auto startPointer = core_size * GetBlockIdx() * block_size;
        this->startPointer = startPointer;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, block_size * this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalLength);

        this->tileNum = this->blockLength;

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->batch * this->bs * this->tileLength * sizeof(T));
        // pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        auto length = this->batch * this->bs * this->tileLength;
        auto length2 = length / this->bs;
        auto d = this->batch * this->bs;

        DataCopyParams params;
        params.blockCount = this->batch;
        params.blockLen = this->tileLength / this->ALIGN_NUM;
        params.srcStride = this->tileLength / this->ALIGN_NUM;
        params.dstStride = 0;
                
        auto mod1 = this->bs - 1;
        auto mod2 = this->shape[2] - 1;
        auto div2 = this->bit[4];
        auto div3 = ~((1 << (this->bit[4] + this->bit[2])) - 1);
        auto mul3 = this->bit[3] - this->bit[4];

        auto st2 = (this->st + d - 1) / d * d - this->st;
        auto ed2 = (loopCount - st2) / d * d + st2;

        for(int32_t i = 0; i < st2; i++) {
            {
                LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                DataCopy(xLocal, xGm[i << mul3], this->tileLength);
                inQueueX.EnQue(xLocal);
            }
            {
                LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                auto j = this->st + i;
                auto hy = j & div3;
                auto x = (j & mod1) << this->bit[2];
                auto w = (j >> div2) & mod2;
                
                DataCopy(zGm[(hy ^ x ^ w) << mul3], xLocal, this->tileLength);
                // free output tensor for reuse
                inQueueX.FreeTensor(xLocal);
            }
        }

        for (int32_t i = st2; i < ed2; i+=d) {
            {
                LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                DataCopy(xLocal, xGm[i << mul3], length);
                inQueueX.EnQue(xLocal);
            }
            {
                LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                auto j = this->st + i;
                auto hy = j & div3;
                auto w = (j >> div2) & mod2;
                auto hyw = (hy ^ w) << mul3;
                
                for(int32_t k = 0; k < this->bs; k++){
                    auto k3 = k << mul3;
                    auto x = k3 << this->bit[2];

                    DataCopy(zGm[hyw ^ x], xLocal[k3], params);
                }
                // free output tensor for reuse
                inQueueX.FreeTensor(xLocal);
            }
        }

        for(int32_t i = ed2; i < loopCount; i++) {
            {
                LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                DataCopy(xLocal, xGm[i << mul3], this->tileLength);
                inQueueX.EnQue(xLocal);
            }
            {
                LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                auto j = this->st + i;
                auto hy = j & div3;
                auto x = (j & mod1) << this->bit[2];
                auto w = (j >> div2) & mod2;
                
                DataCopy(zGm[(hy ^ x ^ w) << mul3], xLocal, this->tileLength);
                // free output tensor for reuse
                inQueueX.FreeTensor(xLocal);
            }
        }
    }
    // __aicore__ inline void Process()
    // {
    //     // loop count need to be doubled, due to double buffer
    //     int32_t loopCount = this->tileNum;
    //     // tiling strategy, pipeline parallel
    //     for (int32_t i = 0; i < loopCount; i++) {
    //         CopyIn(i, this->tileLength);
    //         Compute(i, this->tileLength);
    //         CopyOut(i, this->tileLength);
    //     }
    // }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        DataCopy(zLocal, xLocal, length);

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        // copy progress_th tile from local tensor to global tensor
        // auto mod1 = this->bs - 1;
        // auto mod2 = this->shape[2] - 1;
        // auto div2 = this->bit[4];
        // auto div3 = ~(1 << (this->bit[4] + this->bit[2])) - 1;
        // auto i = this->st + progress;
        // auto hy = i & div3;
        // auto x = (i & mod1) << this->bit[2];
        // auto w = (i >> div2) & mod2;

        auto i = this->st + progress;
        auto hy = i / (this->shape[2] * this->shape[3] / length) * (this->shape[2] * this->shape[3] / length);
        auto x = i % this->bs * this->shape[2];
        auto w = i / this->bs % this->shape[2];

        DataCopy(zGm[(hy ^ x ^ w) * this->tileLength], zLocal, length);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX;
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
    uint32_t ALIGN_NUM;
    TransposeParamsExt transposeParams1, transposeParams2;
    LocalTensor<T> signbit;
};

template<typename T, unsigned opType> class KernelDeepToSpace3 {
public:
    __aicore__ inline KernelDeepToSpace3() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t* shape, uint32_t batch, uint32_t* bit=nullptr)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() / 2 == GetBlockIdx() / 2 + 1 ? core_remain : 0);
        this->bit = bit;
        this->core_size = core_size;

        this->st = core_size * (GetBlockIdx() / 2);
        this->ed = this->st + this->blockLength;

        this->tileLength = block_size;

        this->ALIGN_NUM = ALIGN_NUM;
        this->shape = shape;
        this->batch = batch;

        this->startPointer = core_size * (GetBlockIdx() / 2) * block_size;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + this->startPointer, block_size * this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalLength);

        this->tileNum = this->blockLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->batch * this->tileLength * sizeof(T));
        // pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->batch * this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        auto length = this->batch * this->bs * this->tileLength;
        auto length2 = length / this->bs;
        auto d = this->batch * this->bs;

        DataCopyParams params;
        params.blockCount = this->batch;
        params.blockLen = this->tileLength / this->ALIGN_NUM;
        params.srcStride = this->tileLength / this->ALIGN_NUM;
        params.dstStride = 0;
                
        const auto mod1 = 1;
        auto mod2 = this->shape[2] - 1;
        const auto div2 = 1;
        auto div3 = ~((1 << (1 + this->bit[2])) - 1);
        auto mul3 = this->bit[3] - 1;
        auto k3 = 1 << mul3;
        auto k3x = 1 << (mul3 + this->bit[2]);

        auto st2 = (this->st + d - 1) / d * d - this->st;
        auto ed2 = (loopCount - st2) / d * d + st2;

        if(st2) {
            DataCopyParams params;
            params.blockCount = st2 / 2;
            params.blockLen = this->tileLength / this->ALIGN_NUM;
            params.srcStride = this->tileLength / this->ALIGN_NUM;
            params.dstStride = 0;
            if(GetBlockIdx() & 1){
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm[1 << mul3], params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    auto j = this->st;
                    auto hy = j & div3;
                    auto w = (j >> div2) & mod2;
                    auto hyw = hy ^ w;

                    DataCopy(zGm[(hyw << mul3) ^ k3x], xLocal, this->tileLength * st2 / 2);
                    inQueueX.FreeTensor(xLocal);
                }
            }else{
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm, params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    auto j = this->st;
                    auto hy = j & div3;
                    auto w = (j >> div2) & mod2;
                    auto hyw = hy ^ w;

                    DataCopy(zGm[hyw << mul3], xLocal, this->tileLength * st2 / 2);
                    inQueueX.FreeTensor(xLocal);
                }
            }
            
        }

        auto st3 = this->st << mul3;
        auto ed3 = ed2 << mul3;
        st2 <<= mul3;
        d <<= mul3;
        div3 <<= mul3;
        mod2 <<= mul3;
        auto add3 = 1 << mul3;

        // for (int32_t i = 0; i < ed3; i+=d) {
        //     auto j = st3 + i;
        //     auto hy = j & div3;
        //     auto w = (j >> div2) & mod2;
        //     auto hyw = hy ^ w;
        //     {
        //         LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        //         LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        //         DataCopy(xLocal, xGm[i], params);
        //         DataCopy(xLocal2, xGm[i + add3], params);
        //         inQueueX.EnQue(xLocal);
        //         inQueueX2.EnQue(xLocal2);
        //     }
        //     {
        //         LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        //         LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        //         DataCopy(zGm[hyw], xLocal, length2);
        //         DataCopy(zGm[hyw ^ k3x], xLocal2, length2);
        //         inQueueX.FreeTensor(xLocal);
        //         inQueueX2.FreeTensor(xLocal2);
        //     }
        // }

        
        if(GetBlockIdx() & 1){
            for (int32_t i = st2; i < ed3; i+=d) {
                auto j = st3 + i;
                auto hy = j & div3;
                auto w = (j >> div2) & mod2;
                auto hyw = hy ^ w;
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm[i + add3], params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    DataCopy(zGm[hyw ^ k3x], xLocal, length2);
                    inQueueX.FreeTensor(xLocal);
                }
            }
        }else{
            for (int32_t i = st2; i < ed3; i+=d) {
                auto j = st3 + i;
                auto hy = j & div3;
                auto w = (j >> div2) & mod2;
                auto hyw = hy ^ w;
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm[i], params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    DataCopy(zGm[hyw], xLocal, length2);
                    inQueueX.FreeTensor(xLocal);
                }
            }
        }

        if(ed2 < loopCount) {
            DataCopyParams params;
            params.blockCount = (loopCount - ed2 + 1) / 2;
            params.blockLen = this->tileLength / this->ALIGN_NUM;
            params.srcStride = this->tileLength / this->ALIGN_NUM;
            params.dstStride = 0;
            if(GetBlockIdx() & 1){
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm[ed3 + add3], params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    auto j = st3 + ed3;
                    auto hy = j & div3;
                    auto w = (j >> div2) & mod2;
                    auto hyw = hy ^ w;

                    DataCopy(zGm[hyw ^ k3x], xLocal, this->tileLength * params.blockCount);
                    inQueueX.FreeTensor(xLocal);
                }
            }else{
                {
                    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                    DataCopy(xLocal, xGm[ed3], params);
                    inQueueX.EnQue(xLocal);
                }
                {
                    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                    auto j = st3 + ed3;
                    auto hy = j & div3;
                    auto w = (j >> div2) & mod2;
                    auto hyw = hy ^ w;

                    DataCopy(zGm[hyw], xLocal, this->tileLength * params.blockCount);
                    inQueueX.FreeTensor(xLocal);
                }
            }
        }
    }

private:
    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX;
    // TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    uint32_t blockLength;
    uint32_t core_size;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t startPointer;
    const uint32_t bs = 2;
    uint32_t* shape;
    uint32_t* bit;
    uint32_t batch;
    uint32_t st, ed;
    uint32_t ALIGN_NUM;
};

extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(tiling_data.type == 8){
        KernelDeepToSpace3 <DTYPE_X, 8> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.batch, tiling_data.bit);
        op.Process();
    }else if(tiling_data.type == 7){
        KernelDeepToSpace2 <DTYPE_X, 7> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bs, tiling_data.shape, tiling_data.batch, tiling_data.bit);
        op.Process();
    }else if(tiling_data.type == 0){
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
    }
}
