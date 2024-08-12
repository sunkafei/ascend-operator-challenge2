#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 4;                                     // tensor num for each queue

template<typename T, unsigned opType> class KernelBallQuery {
public:
    __aicore__ inline KernelBallQuery() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR sz1, GM_ADDR sz2, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t bs, uint32_t totalLength2, float min_radius, float max_radius, uint32_t sample_num)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->totalLength2 = totalLength2;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);

        this->st = core_size * GetBlockIdx() / 3;
        this->ed = this->st + this->blockLength / 3;

        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        this->bs = bs;
        this->min_radius = min_radius;
        this->max_radius = max_radius;
        this->sample_num = sample_num;

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;
        this->startPointer = startPointer;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, totalLength);
        yGm.SetGlobalBuffer((__gm__ T*)y, totalLength2);
        zGm.SetGlobalBuffer((__gm__ uint32_t*)z + startPointer / 3 * this->sample_num, (this->ed-this->st)*this->sample_num);
        if constexpr (opType == 1){
            sz1Gm.SetGlobalBuffer((__gm__ uint32_t*)sz1, bs);
            sz2Gm.SetGlobalBuffer((__gm__ uint32_t*)sz2, bs);
        }

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        // pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(uint32_t));
    }
    __aicore__ inline void Process()
    {
        uint32_t N = this->totalLength / this->bs / 3;
        uint32_t N2 = this->totalLength2 / this->bs / 3;
        auto min_radius = this->min_radius * this->min_radius;
        if constexpr (opType == 1) min_radius = 0;
        auto max_radius = this->max_radius * this->max_radius;
        auto sample_num = this->sample_num;
        auto zstartPointer = this->startPointer / 3 * this->sample_num;
        uint32_t st = this->st * this->sample_num - zstartPointer;
        for(uint32_t i=this->st;i<this->ed;i++){
            uint32_t b;
            if constexpr (opType == 1){
                uint32_t s = i;
                b = 0;
                while(b < this->bs && s >= sz1Gm.GetValue(b)){
                    s -= sz1Gm.GetValue(b);
                    b++;
                }
                if(b == this->bs){
                    zGm.SetValue(st, -1);
                    for(int k=1;k<this->sample_num;k++){
                        zGm.SetValue(st + k, 0);
                    }
                    st += this->sample_num;
                    continue;
                }
            }else{
                b = i / N;
            }
            float x = xGm.GetValue(i * 3 - this->startPointer);
            float y = xGm.GetValue(i * 3 + 1 - this->startPointer);
            float z = xGm.GetValue(i * 3 + 2 - this->startPointer);
            uint32_t st2 = 0;
            uint32_t ed2 = 0;
            if constexpr (opType == 1){
                for(int j=0;j<b;j++){
                    st2 += sz2Gm.GetValue(j);
                }
                ed2 = st2 + sz2Gm.GetValue(b);
            }else{
                st2 = b * N2;
                ed2 = st2 + N2;
            }
            uint32_t cnt = 0;
            uint32_t lst = 0;
            for(uint32_t j=st2,n=0;j<ed2;j++,n++){
                float a = yGm.GetValue(j * 3);
                float b = yGm.GetValue(j * 3 + 1);
                float c = yGm.GetValue(j * 3 + 2);
                float tmp = (x - a) * (x - a) + (y - b) * (y - b) + (z - c) * (z - c);
                if(tmp == 0 || (min_radius <= tmp && tmp < max_radius)){
                    zGm.SetValue(st + cnt, n);
                    lst = n;
                    cnt++;
                    if(cnt == this->sample_num) break;
                }
            }
            if(cnt != this->sample_num){
                for(int k=cnt;k<this->sample_num;k++){
                    zGm.SetValue(st + k, lst);
                }
            }
            st += this->sample_num;
        }
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(zGm);
        
        // // loop count need to be doubled, due to double buffer
        // int32_t loopCount = this->tileNum;
        // // tiling strategy, pipeline parallel
        // for (int32_t i = 0; i < loopCount-1; i++) {
        //     // CopyIn(i, this->tileLength);
        //     Compute(i, this->tileLength);
        //     CopyOut(i, this->tileLength);
        // }
        // auto length = this->ed - this->st - this->tileLength * (loopCount - 1);
        // // CopyIn(loopCount - 1, length);
        // Compute(loopCount - 1, length);
        // CopyOut(loopCount - 1, length);
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
        // deque input tensors from VECIN queue
        // LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint32_t> zLocal = outQueueZ.AllocTensor<T>();

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<uint32_t>(zLocal);
        // free input tensors for reuse
        // inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<uint32_t> zLocal = outQueueZ.DeQue<T>();
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
    GlobalTensor<T> yGm;
    GlobalTensor<uint32_t> sz1Gm;
    GlobalTensor<uint32_t> sz2Gm;
    GlobalTensor<uint32_t> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t totalLength2;
    uint32_t startPointer;
    uint32_t bs;
    uint32_t st, ed;
    float min_radius;
    float max_radius;
    uint32_t sample_num;
    LocalTensor<T> signbit;
};

extern "C" __global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(tiling_data.opType == 0){
        KernelBallQuery <DTYPE_XYZ, 0> op;
        op.Init(center_xyz, xyz, center_xyz_batch, xyz_batch_cnt, idx, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bDim, tiling_data.totalLength2, tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num);
        op.Process();
    }else if(tiling_data.opType == 1){
        KernelBallQuery <DTYPE_XYZ, 1> op;
        op.Init(center_xyz, xyz, center_xyz_batch, xyz_batch_cnt, idx, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.bDim, tiling_data.totalLength2, tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num);
        op.Process();
    }
}
