"""Compare two [n, dim] npy vector dumps: max abs diff and min / median cosine (rows that are zero in both skipped)."""
import json, sys
import numpy as np
a, b = (np.nan_to_num(np.load(x).astype(np.float64)) for x in sys.argv[1:3])
na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
live = (na > 0) | (nb > 0)
cos = (a[live] * b[live]).sum(1) / np.maximum(na[live] * nb[live], 1e-30)
print(json.dumps({"a": sys.argv[1], "b": sys.argv[2], "rows": int(len(a)), "zero_rows": int((~live).sum()),
                  "max_abs_diff": float(np.abs(a - b).max()), "min_cos": float(cos.min()), "median_cos": float(np.median(cos))}))
