
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BallQueryTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength2);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, core_size);
    TILING_DATA_FIELD_DEF(uint32_t, core_remain);
    TILING_DATA_FIELD_DEF(uint32_t, opType);
    TILING_DATA_FIELD_DEF(uint32_t, bDim);
    TILING_DATA_FIELD_DEF(float, min_radius);
    TILING_DATA_FIELD_DEF(float, max_radius);
    TILING_DATA_FIELD_DEF(uint32_t, sample_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BallQuery, BallQueryTilingData)
}
