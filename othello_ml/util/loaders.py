import numpy as np
import pyzstd
import torch
import time

from multiprocessing import Process, Queue

# hack around pytorch dataloader being slow for tensors in ram
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


N_PROC = 2

def _fill(input_dim, q, batches, proc_idx):
    open_filename = None
    data          = None

    for batch_idx, batch_entries in enumerate(batches):
        # round robining the batches does not keep contig files read on the same
        # process, so we need to try keeping ~8 contig batches on the same
        # process

        if (batch_idx // 8) % N_PROC != proc_idx:
            continue

        batch_size = len(batch_entries)
        batch      = np.zeros( (batch_size, input_dim) )

        for entry_idx, entry in enumerate(batch_entries):
            file_name = entry[0]
            file_idx  = entry[1]

            if file_name != open_filename:
                with open(file_name, 'rb') as f:
                    decompressed_data = pyzstd.decompress(f.read())
                    buf = np.frombuffer(decompressed_data, dtype=np.float32)

                    open_filename = file_name
                    data = buf.reshape( (len(buf) // input_dim, input_dim) )

            batch[entry_idx,:] = data[file_idx,:]

        q.put( (batch_idx, batch) ) # blocks if full

    q.put(None) # sentinal for done

class EntryIterator(object):
    def __init__(self, input_dim, batches):
        self.nxt = 0

        self.procs = []
        self.qs    = []
        self.done  = 0

        for idx in range(0, N_PROC):
            self.qs.append(Queue(1024))
            self.procs.append(Process(target=_fill, args=(input_dim, self.qs[idx], batches, idx)))

        for proc in self.procs:
            proc.daemon = True # makes it easier to deal wih not actually cleaning up
            proc.start()

    def __next__(self):
        # figure out which queue we need to look into next
        q_idx = (self.nxt // 8) % N_PROC
        q = self.qs[q_idx]

        # we're done when we hit the _first_ done
        # because we always know which queue to target
        r = q.get()
        if r is None:
            # join all of the processes
            for proc in self.procs:
                proc.join()
                proc.close()

            return None

        i, batch = r
        assert i == self.nxt, f'expected {self.nxt} but got {i}, popped from queue {q_idx}'

        self.nxt += 1

        return torch.Tensor(batch)


class BoardDirLoader(object):
    """
    loads board data from the split up board files we have to produce.
    not enough ram to while entire dataset in memory for the large lookback models
    """

    def __init__(self, boards_dir, boards_per_file, input_dim, indices, batch_size):
        """
        boards_dir: directory with boards files in it
        boards_per_file: how many entries make it into each boards file
        input_dim: dimensions of each entry
        indicies: indicies to fetch from the files, in the order they are requested
        """

        self.input_dim = input_dim

        # compute the file_name and file_index we'll need for each entry
        entries = []
        for idx in indices:
            file_num = idx // boards_per_file
            file_idx = idx - (boards_per_file * file_num)
            file_name = f'{boards_dir}/{file_num:05}.dat.zst'

            entries.append( (file_name, file_idx) )

        # now collapse those into batches
        self.batches = []

        for i in range(0, len(entries) // batch_size +1):
            st = i * batch_size
            ed = min((i+1) * batch_size, len(entries))

            if ed <= st:
                break

            self.batches.append(entries[st:ed])

        # subprocesses created in the pool and not consumed will never get
        # properly joined or closed
        self.iter_pool = []
        for _ in range(2):
            self.iter_pool.append(EntryIterator(self.input_dim, self.batches))

    def __iter__(self):
        ret = self.iter_pool.pop()
        self.iter_pool.append(EntryIterator(self.input_dim, self.batches))
        return ret
