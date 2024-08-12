
#include "ball_query_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    BallQueryTilingData tiling;
    constexpr int32_t NUM = 4;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    // std::cout << aivNum << " " << ub_size << " " << ascendcPlatform.GetCoreNumAic() << " " << ascendcPlatform.GetCoreNumAiv() << std::endl;

    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
    }else{
        sizeofdatatype = 4;
    }
    // std::cout << mode << " " << format << " " << type << std::endl;

    uint32_t totalLength = context->GetInputTensor(1)->GetShapeSize();
    uint32_t totalLength2 = context->GetInputTensor(0)->GetShapeSize();

    if(context->GetInputTensor(2) == nullptr){
        const gert::Shape vec = context->GetInputShape(0)->GetOriginShape();
        tiling.set_opType(0);
        tiling.set_bDim(vec.GetDim(0));
    }else{
        tiling.set_opType(1);
        auto b = context->GetInputTensor(2)->GetShapeSize() < context->GetInputTensor(1)->GetShapeSize() ? context->GetInputTensor(2)->GetShapeSize() : context->GetInputTensor(1)->GetShapeSize();
        tiling.set_bDim(b);
    }
    tiling.set_min_radius(*context->GetAttrs()->GetFloat(0));
    tiling.set_max_radius(*context->GetAttrs()->GetFloat(1));
    tiling.set_sample_num(*context->GetAttrs()->GetInt(2));

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;
    //aivNum = 1;

    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 24) * (ALIGN_NUM * 24);
    uint32_t core_remain = totalLength - aivNum * core_size;

    tiling.set_totalLength(totalLength);
    tiling.set_totalLength2(totalLength2);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    // std::cout << aivNum << " " << core_size << " " << core_remain << " " << totalLength << std::endl;
    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class BallQuery : public OpDef {
public:
    explicit BallQuery(const char* name) : OpDef(name)
    {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz_batch")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min_radius").Float();
        this->Attr("max_radius").Float();
        this->Attr("sample_num").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(BallQuery);
}
