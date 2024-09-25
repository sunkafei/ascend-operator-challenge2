
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    PdistTilingData tiling;
    auto p = *context->GetAttrs()->GetFloat(0);
    tiling.set_p(p);
    auto n = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    tiling.set_n(n);
    auto m = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    tiling.set_m(m);
    int bits = 0;
    for (int i = 0; i < 31; ++i) if (m & (1 << i)) {
        bits += 1;
    }
    tiling.set_single_bits(bits == 1 && m >= 32);

    std::cout << "p: " << p << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "m: " << m << std::endl;
    std::cout << "bits: " << bits << std::endl;
    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    std::cout << "ub_size: " << ub_size << std::endl;

    uint32_t aivNum = 40;
    uint32_t core_size = (n + 1) / 2 / aivNum;
    if(core_size == 0){
        aivNum = (n + 1) / 2;
    }
    uint32_t core_remain = (n + 1) / 2 - aivNum * core_size;

    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Pdist);
}
