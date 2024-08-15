
#include "ball_query_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    BallQueryTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto num_cores = 1;
    int32_t sizeofdatatype;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        sizeofdatatype = 2;
    }
    else {
        sizeofdatatype = 4;
    }
    auto min_radius = *context->GetAttrs()->GetFloat(0);
    tiling.set_min_radius(min_radius);
    auto max_radius = *context->GetAttrs()->GetFloat(1);
    tiling.set_max_radius(max_radius);
    auto sample_num = *context->GetAttrs()->GetInt(2);
    tiling.set_sample_num(sample_num);
    int32_t type = 0;
    auto batch_size = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto num_points = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto num_centers = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    if (context->GetInputTensor(2) != nullptr) {
        type = 1;
        batch_size = context->GetInputShape(2)->GetStorageShape().GetDim(0);
        num_points = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        num_centers = context->GetInputShape(1)->GetStorageShape().GetDim(0);
        if (context->GetInputShape(2)->GetStorageShape().GetDim(0) != context->GetInputShape(3)->GetStorageShape().GetDim(0)) {
            abort(); //todo
        }
    }
    else if (num_points * sizeofdatatype % 64 == 0 || num_centers * batch_size < 40000) {
        type = -1;
        num_cores = ascendcPlatform.GetCoreNum();
    }
    tiling.set_type(type);
    tiling.set_batch_size(batch_size);
    tiling.set_num_points(num_points);
    tiling.set_num_centers(num_centers);

    std::cerr << "type: " << type << std::endl;
    std::cerr << "batch_size: " << batch_size << std::endl;
    std::cerr << "num_points: " << num_points << std::endl;
    std::cerr << "num_centers: " << num_centers << std::endl;
    std::cerr << "min_radius: " << min_radius << std::endl;
    std::cerr << "max_radius: " << max_radius << std::endl;
    std::cerr << "sample_num: " << sample_num << std::endl;
    std::cerr << "num_cores: " << num_cores << std::endl;


    context->SetBlockDim(num_cores);
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
