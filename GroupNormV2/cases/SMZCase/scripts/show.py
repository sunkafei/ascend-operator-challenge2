import numpy as np
import sys
mean = np.fromfile("/home/lhq/ascend-operator-challenge2/GroupNormV2/cases/SMZCase/output/mean.bin", dtype=np.float16)
print("mean:", mean, file=sys.stderr)
rstd = np.fromfile("/home/lhq/ascend-operator-challenge2/GroupNormV2/cases/SMZCase/output/rstd.bin", dtype=np.float16)
print("rstd:", rstd, file=sys.stderr)
#output = np.fromfile("/home/lhq/ascend-operator-challenge2/GroupNormV2/cases/SMZCase/output/output.bin", dtype=np.float16)
#print("output:", output[:8], file=sys.stderr)