
#include "group_norm_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <iostream>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GroupNormV2TilingData tiling;

    auto num_groups = *context->GetAttrs()->GetInt(0);
    tiling.set_num_groups(num_groups);
    auto shape = context->GetInputShape(0)->GetStorageShape();
    auto batch_size = shape.GetDim(0);
    tiling.set_batch_size(batch_size);
    auto num_channels = shape.GetDim(1);
    tiling.set_num_channels(num_channels);
    auto total_size = context->GetInputTensor(0)->GetShapeSize();
    tiling.set_total_size(total_size);
    auto epsilon = *context->GetAttrs()->GetFloat(2);
    tiling.set_epsilon(epsilon);
    int32_t dimension = total_size / batch_size / num_groups;
    int32_t chunk_size = std::ceil(std::sqrt(dimension));
    while (dimension % chunk_size) {
        chunk_size -= 1;
    }
    tiling.set_chunk_size(chunk_size);
    int32_t sizeofdatatype;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        sizeofdatatype = 2;
    }
    else {
        sizeofdatatype = 4;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto num_cores = ascendcPlatform.GetCoreNum();
    if (total_size / batch_size / num_groups % (64 / sizeofdatatype) != 0) {
        num_cores = 1;
    }

    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    const auto limit = total_size / batch_size / num_channels;
    auto tile_length = -1;
    auto temp_length = -1;
    for (int i = 1; i < limit; i += 1) {
        if (limit % i != 0) {
            continue;
        }
        auto size = limit / i;
        if (size * sizeofdatatype % 32 != 0) {
            continue;
        }
        uint32_t min, max;
        AscendC::GetSumMaxMinTmpSize(size, sizeofdatatype, false, min, max);
        if ((size + max) * sizeofdatatype * 4 > ub_size * 0.9) {
            continue;
        }
        tile_length = size;
        temp_length = max;
        break;
    }
    auto span = (total_size / tile_length - 1) / num_cores + 1;
    if (sizeofdatatype == 2) { // todo:提升fp16的处理精度
        tile_length = -1;
    }
    if (tile_length == -1) {
        span = (batch_size * num_groups - 1) / num_cores + 1;
    }
    tiling.set_temp_length(temp_length);
    tiling.set_tile_length(tile_length);
    tiling.set_span(span);

    context->SetBlockDim(num_cores);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    std::cerr << "ub_size: " << ub_size << std::endl;
    std::cerr << "chunk_size: " << chunk_size << std::endl;
    std::cerr << "num_groups: " << num_groups << std::endl;
    std::cerr << "batch_size: " << batch_size << std::endl;
    std::cerr << "num_channels: " << num_channels << std::endl;
    std::cerr << "total_size: " << total_size << std::endl;
    std::cerr << "epsilon: " << epsilon << std::endl;
    std::cerr << "num_cores: " << num_cores << std::endl;
    std::cerr << "span: " << span << std::endl;
    std::cerr << "tile_length: " << tile_length << std::endl;

    size_t usrSize = 1024 * 1024 * 40; // 设置用户需要使用的workspace大小。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GroupNormV2 : public OpDef {
public:
    explicit GroupNormV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_groups").Int();
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("eps").AttrType(OPTIONAL).Float(0.0001);
        this->Attr("is_training").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(GroupNormV2);
}
