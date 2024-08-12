
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BallQueryTilingData)
    TILING_DATA_FIELD_DEF(int32_t, type);
    TILING_DATA_FIELD_DEF(int32_t, batch_size);
    TILING_DATA_FIELD_DEF(int32_t, num_centers);
    TILING_DATA_FIELD_DEF(int32_t, num_points);
    TILING_DATA_FIELD_DEF(float, min_radius);
    TILING_DATA_FIELD_DEF(float, max_radius);
    TILING_DATA_FIELD_DEF(int32_t, sample_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BallQuery, BallQueryTilingData)
}
