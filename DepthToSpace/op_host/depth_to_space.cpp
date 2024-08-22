
#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    DepthToSpaceTilingData tiling;
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

    tiling.set_bs(*context->GetAttrs()->GetInt(0));
    const gert::Shape vec = context->GetInputShape(0)->GetOriginShape();
    uint32_t shape[4] = {0};
    uint32_t bit[4] = {0};
    for(int i=0;i<vec.GetDimNum();i++){
        shape[i] = vec.GetDim(i);
        bit[i] = __builtin_ctz(shape[i]);
    }
    bit[4] = __builtin_ctz(*context->GetAttrs()->GetInt(0));
    tiling.set_shape(shape);
    tiling.set_bit(bit);
    const char *mode = context->GetAttrs()->GetAttrPointer<char>(1);
    const char *format = context->GetAttrs()->GetAttrPointer<char>(2);

    uint32_t type = 0;
    if(strcmp(mode, "CRD") == 0){
        type |= 1;
    }
    if(strcmp(format, "NHWC") == 0){
        type |= 2;
    }
    if(type == 2){
        if(__builtin_popcount(shape[2]) == 1 && __builtin_popcount(shape[3]) == 1){
            type = 4;
            if(shape[2] > BLOCK_SIZE / sizeofdatatype && shape[3] / *context->GetAttrs()->GetInt(0) > BLOCK_SIZE / sizeofdatatype){
                type = 7;
            }
        }else if(__builtin_popcount(shape[2]) == 1){
            type = 5;
        }else if(__builtin_popcount(shape[3]) == 1){
            type = 6;
        }
    }
    tiling.set_type(type);
    // std::cout << mode << " " << format << " " << type << std::endl;

    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;
    // aivNum = 1;
    aivNum = ascendcPlatform.GetCoreNum();

    // uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_size = (totalLength / aivNum + (ALIGN_NUM * 8) - 1) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    aivNum = (totalLength + core_size - 1) / core_size;
    uint32_t core_remain = totalLength - aivNum * core_size;

    if(type == 7){
        if(*context->GetAttrs()->GetInt(0) == 2) type = 8;
        tiling_size = ((ub_size) / BLOCK_SIZE / 1) / NUM;
        block_size = tiling_size * ALIGN_NUM;
        block_size = block_size / shape[3] * shape[3];
        while(shape[2] % (block_size / shape[3])){
            block_size -= shape[3];
        }
        auto batch = block_size / shape[3];
        tiling.set_batch(batch);

        block_size = shape[3] / *context->GetAttrs()->GetInt(0);
        auto n = totalLength / block_size;
        aivNum = ascendcPlatform.GetCoreNum() * 11;
        core_size = (n + aivNum - 1) / aivNum;
        aivNum = (n + core_size - 1) / core_size;
        core_remain = n - aivNum * core_size;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    // std::cout << aivNum << " " << type << " " << core_size << " " << core_remain << " " << core_size + core_remain << " " << totalLength << std::endl;
    context->SetBlockDim(aivNum);

    // auto bs = *context->GetAttrs()->GetInt(0);
    // auto div1 = shape[1] * shape[2] * shape[3];
    // auto div2 = shape[2] * shape[3];
    // auto div3 = shape[3];
    // auto div4 = shape[3] / bs;
    // auto div5 = shape[3] / bs / bs;

    // auto mul1 = shape[1] * shape[2] * shape[3];
    // auto mul2 = shape[2] * shape[3];
    // auto mul3 = shape[2] * bs * div5;
    // auto mul4 = bs * div5;
    // auto mul5 = div5;
    // for(uint32_t i=0;i<3*64*64;i++){
    //     auto b = i / div1;
    //     auto h = i / div2 % shape[1];
    //     auto w = i / div3 % shape[2];
    //     auto x = i / div4 % bs;
    //     auto y = i / div5 % bs;
    //     auto c = i % div5;

    //     std::cout << i << " " << b << " " << h << " " << x << " " << w << " " << y << " " << c << " " << b * mul1 + h * mul2 + x * mul3 + w * mul4 + y * mul5 + c << std::endl;
    // }

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
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").Int();
        this->Attr("mode").AttrType(OPTIONAL).String("DCR");
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(DepthToSpace);
}
