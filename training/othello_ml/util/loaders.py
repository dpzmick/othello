import bisect
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

            # data_gen_policy.c writes uint8 to disk (input planes are all
            # 0/1). The Python side reads it back as uint8, indexes by the
            # train/test row indices, and stores the subset. The per-batch
            # upcast to float32 happens later in __iter__.
            indices_sorted = np.asarray(indices)
            assert (indices_sorted[1:] >= indices_sorted[:-1]).all(), \
                "indices must be sorted ascending (make_split produces these via np.where)"

            out = np.empty((len(indices_sorted), input_dim), dtype=np.uint8)
            write_pos = 0
            running_offset = 0

            for path in files:
                with open(path, 'rb') as f:
                    decomp = pyzstd.decompress(f.read())
                arr = np.frombuffer(decomp, dtype=np.uint8).reshape(-1, input_dim)
                file_end = running_offset + arr.shape[0]

                # Slice of the global indices that fall within this file.
                lo = bisect.bisect_left(indices_sorted, running_offset)
                hi = bisect.bisect_left(indices_sorted, file_end)
                if hi > lo:
                    local_idx = indices_sorted[lo:hi] - running_offset
                    out[write_pos:write_pos + (hi - lo)] = arr[local_idx]
                    write_pos += (hi - lo)

                running_offset = file_end
                # `arr` goes out of scope here so the file's buffer is freed
                # before the next iteration.

            assert write_pos == len(indices_sorted), \
                f"indexed {write_pos} rows but expected {len(indices_sorted)}"

            # Storage stays in CPU RAM as uint8 (~12 GB for full WTHOR D4).
            # On CUDA we pin the host buffer so per-batch HtoD copies bypass
            # the synchronous staging step and can overlap with compute via
            # non_blocking=True. MPS/CPU get no benefit so we skip pinning
            # there (pinning all 12GB on a 16GB laptop is bad news).
            self.data = torch.from_numpy(out)
            self.device = device
            self._pinned = False
            dev_str = str(device) if not isinstance(device, torch.device) else device.type
            if dev_str.startswith("cuda"):
                self.data = self.data.pin_memory()
                self._pinned = True

    def __iter__(self):
        # uint8 -> float32 + host-to-device move per batch. Pinned source
        # buffer lets the copy run async; .float() then happens GPU-side on
        # the freshly-arrived 3 MB uint8 batch (12 MB float32 temp).
        n = self.data.shape[0]
        non_blocking = self._pinned
        for st in range(0, n, self.batch_size):
            yield self.data[st:st+self.batch_size].to(self.device, non_blocking=non_blocking).float()

    def __len__(self):
        n = self.data.shape[0]
        return (n + self.batch_size - 1) // self.batch_size
