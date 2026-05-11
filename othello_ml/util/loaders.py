import numpy as np
import os
import pyzstd
import torch


# A small in-memory batch slicer used for the policy targets (already
# in-memory tensors at this point in the pipeline).
class SimpleDataLoader(object):
    def __init__(self, dset, batch_size):
        iterable = []
        for i in range(0, int(len(dset)/batch_size)+1):
            st = i * batch_size
            ed = min((i+1) * batch_size, len(dset))

            if ed <= st:
                break

            x = dset[st:ed]
            iterable.append(x)

        self._inner = iterable

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


class BoardDirLoader(object):
    """Loads policy training inputs from the zstd-compressed board files
    produced by data_gen_policy.c.

    Decompresses every file once, concatenates into a single numpy array,
    selects the (train or test) subset by index, then yields batches by
    slicing. Works as long as the dataset fits in RAM. Without lookback,
    that bound is roughly 3M rows * 193 floats * 4 bytes ~= 2.4 GB, fine
    on a 16 GB machine.

    Older versions of this file had a multiprocessing prefetcher that used
    /dev/shm + fallocate to overlap zstd decompression with training. That
    was justified for the lookback=5 experiments (~20 GB datasets), but for
    everything else it's unnecessary and Linux-only. If we ever genuinely
    need streaming back, the right replacement is stdlib
    multiprocessing.shared_memory rather than hand-rolled /dev/shm.
    """

    def __init__(self, which, boards_dir, boards_per_file, input_dim, indices, batch_size, tracer, device='cpu'):
        del boards_per_file  # implicit in the file rollover, we don't need it here
        self.batch_size = batch_size

        with tracer.start(f"load_{which}_boards"):
            files = sorted(
                os.path.join(boards_dir, p)
                for p in os.listdir(boards_dir)
                if p.endswith('.dat.zst')
            )

            chunks = []
            for path in files:
                with open(path, 'rb') as f:
                    data = pyzstd.decompress(f.read())
                arr = np.frombuffer(data, dtype=np.float32).reshape(-1, input_dim)
                chunks.append(arr)

            all_data = np.concatenate(chunks, axis=0)
            subset = all_data[indices].copy()  # .copy() so torch owns the buffer
            # Move once at startup so the per-batch loop doesn't pay a host->device
            # copy. On M1 the unified memory means this is cheap and the train+test
            # data (a few hundred MB) easily fits.
            self.data = torch.from_numpy(subset).to(device)

    def __iter__(self):
        n = self.data.shape[0]
        for st in range(0, n, self.batch_size):
            yield self.data[st:st+self.batch_size]

    def __len__(self):
        n = self.data.shape[0]
        return (n + self.batch_size - 1) // self.batch_size
