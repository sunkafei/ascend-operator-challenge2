
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  PdistTilingData tiling;
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

    tiling.set_p(*context->GetAttrs()->GetFloat(0));
    const gert::Shape vec = context->GetInputShape(0)->GetOriginShape();
    uint32_t shape[2] = {0};
    for(int i=0;i<vec.GetDimNum();i++){
        shape[i] = vec.GetDim(i);
    }
    tiling.set_shape(shape);

    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;
    // aivNum = 1;
    aivNum = ascendcPlatform.GetCoreNum();
    aivNum = 1;//40;

    // uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_size = shape[0] / aivNum;
    if(core_size == 0){
        aivNum = shape[0];
    }
    uint32_t core_remain = shape[0] - aivNum * core_size;

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    // for(int i = 0; i < core_size; i++){
    //     for(int j = i + 1; j < shape[0]; j++){
    //         std::cout << (2 * shape[0] - 2 - i + 1) * i / 2 + j - i - 1 << " ";
    //     }
    //     std::cout << "---" << (2 * shape[0] - 2 - i + 1) * i / 2 << std::endl;
    // }

    // std::cout << aivNum << " " << core_size << " " << core_remain << " " << core_size + core_remain << " " << totalLength << std::endl;
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
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").Float();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(Pdist);
}
