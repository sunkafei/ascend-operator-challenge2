
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, core_size);
    TILING_DATA_FIELD_DEF(uint32_t, core_remain);
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, shape);
    TILING_DATA_FIELD_DEF(uint32_t, type);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
