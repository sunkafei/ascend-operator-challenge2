
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormV2TilingData)
  TILING_DATA_FIELD_DEF(int32_t, span);
  TILING_DATA_FIELD_DEF(int32_t, chunk_size);
  TILING_DATA_FIELD_DEF(int32_t, batch_size);
  TILING_DATA_FIELD_DEF(int32_t, num_groups);
  TILING_DATA_FIELD_DEF(int32_t, num_channels);
  TILING_DATA_FIELD_DEF(int32_t, total_size);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormV2, GroupNormV2TilingData)
}
