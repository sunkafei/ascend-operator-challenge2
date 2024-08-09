
#include "group_norm_v2_tiling.h"
#include "register/op_def_registry.h"
#include <iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GroupNormV2TilingData tiling;

    auto num_groups_ptr = context->GetAttrs()->GetInt(0);
    tiling.set_num_groups(*num_groups_ptr);
    auto shape = context->GetInputShape(0)->GetStorageShape();
    auto batch_size = shape.GetDim(0);
    tiling.set_batch_size(batch_size);
    auto num_channels = shape.GetDim(1);
    tiling.set_num_channels(num_channels);
    auto total_size = context->GetInputTensor(0)->GetShapeSize();
    tiling.set_total_size(total_size);
    auto epsilon_ptr = context->GetAttrs()->GetFloat(2);
    tiling.set_epsilon(*epsilon_ptr);
    std::cerr << "num_groups: " << *num_groups_ptr << std::endl;
    std::cerr << "batch_size: " << batch_size << std::endl;
    std::cerr << "num_channels: " << num_channels << std::endl;
    std::cerr << "total_size: " << total_size << std::endl;
    std::cerr << "epsilon: " << *epsilon_ptr << std::endl;

    context->SetBlockDim(1);
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