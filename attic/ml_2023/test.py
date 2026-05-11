import numpy as np

ids = np.fromfile("../ids.dat", dtype=np.uint64)
boards = np.fromfile("../boards.dat", dtype=np.float32).reshape( (len(ids), 128) )
policy = np.fromfile("../policy.dat", dtype=np.float32).reshape( (len(ids), 64) )

print(len(ids))
print(boards.shape)
print(policy.shape)
