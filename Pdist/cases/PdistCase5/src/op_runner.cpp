/**
* @file op_runner.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "op_runner.h"
#include "aclnn_pdist.h"
#include <limits>
#include <cassert>
#include "acl/acl_op_compiler.h"
#include "common.h"

using namespace std;

extern bool g_isDevice;

OpRunner::OpRunner(OperatorDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
}

OpRunner::~OpRunner()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto ret = aclDestroyTensor(inputTensor_[i]);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free InputTensor[%d]error code is %d",  static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        ret = aclDestroyDataBuffer(inputBuffers_[i]);

        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free inputBuffers[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        ret = aclrtFree(devInputs_[i]);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free devInputs[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        if (g_isDevice) {
            ret = aclrtFree(hostInputs_[i]);
        } else {
            ret = aclrtFreeHost(hostInputs_[i]);
        }
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free hostInputs[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto ret = aclDestroyTensor(outputTensor_[i]);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free outputTensor[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        ret = aclDestroyDataBuffer(outputBuffers_[i]);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free outputBuffers[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        ret = aclrtFree(devOutputs_[i]);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free devOutputs[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
        if (g_isDevice) {
            ret = aclrtFree(hostOutputs_[i]);
        } else {
            ret = aclrtFreeHost(hostOutputs_[i]);
        }
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Free hostOutputs[%d]error code is %d", static_cast<int32_t>(i), static_cast<int32_t>(ret));
            exit(EXIT_FAILURE);
        }
    }
}

bool OpRunner::Init()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return false;
        }
        devInputs_.emplace_back(devMem);
        inputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostInput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostInput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostInput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        }
        if (hostInput == nullptr) {
            ERROR_LOG("Malloc memory for input[%zu] failed", i);
            return false;
        }
        hostInputs_.emplace_back(hostInput);

        aclTensor *inputTensor = aclCreateTensor(GetInputShape(i).data(), GetInputNumDims(i), GetInputDataType(i),
            nullptr, 0, GetInputFormat(i), GetInputShape(i).data(), GetInputNumDims(i), devInputs_[i]);
        if (inputTensor == nullptr) {
            ERROR_LOG("Create Tensor for input[%zu] failed", i);
            return false;
        }
        inputTensor_.emplace_back(inputTensor);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostOutput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostOutput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return false;
        }
        hostOutputs_.emplace_back(hostOutput);

        aclTensor *outputTensor = aclCreateTensor(GetOutputShape(i).data(), GetOutputNumDims(i), GetOutputDataType(i),
            nullptr, 0, GetOutputFormat(i), GetOutputShape(i).data(), GetOutputNumDims(i), devOutputs_[i]);
        if (outputTensor == nullptr) {
            ERROR_LOG("Create Tensor for output[%zu] failed", i);
            return false;
        }
        outputTensor_.emplace_back(outputTensor);
    }

    return true;
}

const size_t OpRunner::NumInputs()
{
    return numInputs_;
}

const size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

const size_t OpRunner::GetInputNumDims(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->inputDesc[index]);
}

aclDataType OpRunner::GetInputDataType(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->inputDesc[index]);
}

aclFormat OpRunner::GetInputFormat(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ret;
    }

    auto desc = opDesc_->inputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }

    return ret;
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

const size_t OpRunner::GetOutputNumDims(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->outputDesc[index]);
}

aclDataType OpRunner::GetOutputDataType(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->outputDesc[index]);
}


aclFormat OpRunner::GetOutputFormat(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->outputDesc[index]);
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ret;
    }

    auto desc = opDesc_->outputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

bool OpRunner::RunOp()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != ACL_SUCCESS) {
            ERROR_LOG("Copy input[%zu] failed", i);
            return false;
        }
        INFO_LOG("Copy input[%zu] success", i);
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    size_t workspaceSize = 0;
	aclOpExecutor *handle = nullptr;
    
	auto ret = aclnnPdistGetWorkspaceSize(inputTensor_[0],opDesc_->p, outputTensor_[0], &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
	INFO_LOG("Execute GetWorkspaceSize success, workspace size %lu", workspaceSize);
    
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
        }
    }
    ret = aclnnPdist(workspace, workspaceSize, handle, stream);

    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
	INFO_LOG("Execute Operator success");

    ret = aclrtSynchronizeStreamWithTimeout(stream, 5000);
    if (ret != SUCCESS) {
        ERROR_LOG("Synchronize stream failed. error code is %d", static_cast<int32_t>(ret));
        (void)aclrtDestroyStream(stream);
        return false;
    }
    INFO_LOG("Synchronize stream success");

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != ACL_SUCCESS) {
            INFO_LOG("Copy output[%zu] success", i);
            (void)aclrtDestroyStream(stream);
            return false;
        }
        INFO_LOG("Copy output[%zu] success", i);
    }

    (void)aclrtDestroyStream(stream);
    return true;
}


template<typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintFp16Data(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << std::setprecision(4) << aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, aclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case ACL_BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case ACL_INT8:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT8:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT16:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT16:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT32:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT32:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT64:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT64:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT16:
            DoPrintFp16Data(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case ACL_DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}
