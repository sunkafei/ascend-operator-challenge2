
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistTilingData)
  TILING_DATA_FIELD_DEF(float, p);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(int32_t, single_bits);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}
