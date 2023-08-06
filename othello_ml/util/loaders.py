import numpy as np
import pyzstd
import torch
import time
import os
import fallocate
import mmap

from multiprocessing import Process, Queue

# built in shared memory is very annoying
# see: https://bugs.python.org/issue38119
def create_shm(name, size):
    with open(f"/dev/shm/{name}", "wb+") as f:
        fallocate.fallocate(f, 0, size) # noop if already sized appropriately
        # if "batch_0" in name:
        #     print(f'create name={name} sz={size}')
        return mmap.mmap(f.fileno(), size)

def open_shm(name):
    with open(f"/dev/shm/{name}", "rb") as f: # need to be write for default mmap flags
        sz = os.fstat(f.fileno()).st_size
        ret = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        # if "batch_0" in name:
        #     print(f'open/unlink name={name} sz={sz}')

    # don't need this anymore, since we only have one consumer
    os.unlink(f"/dev/shm/{name}")
    return ret

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


def _fill(which, input_dim, q, batches):
    open_filename = None
    data          = None

    for batch_idx, batch_entries in enumerate(batches):
        batch_size = len(batch_entries)
        dim        = (batch_size, input_dim)

        st = time.time()

        # allocate new shm on each request because it's easier
        shm_name = f'{which}_batch_{batch_idx}'
        shm = create_shm(shm_name, dim[0]*dim[1]*8)
        batch = np.ndarray(dim, dtype=np.int64, buffer=shm)

        file_st = None
        file_ed = None

        for entry_idx, entry in enumerate(batch_entries):
            file_name = entry[0]
            file_idx  = entry[1]

            if file_name != open_filename:
                file_st = time.time()
                with open(file_name, 'rb') as f:
                    decompressed_data = pyzstd.decompress(f.read())
                    buf = np.frombuffer(decompressed_data, dtype=np.float32)

                    open_filename = file_name
                    data = buf.reshape( (len(buf) // input_dim, input_dim) )
                file_ed = time.time()

            batch[entry_idx,:] = data[file_idx,:]

        ed = time.time()

        # do not flush, that will mess with file size
        shm.close()

        q.put( (batch_idx, shm_name, (st, ed), (file_st, file_ed)) )

    q.put(None) # sentinal for done

class EntryIterator(object):
    def __init__(self, which, input_dim, batches, tracer):
        self.nxt = 0
        self.tracer = tracer
        self.input_dim = input_dim
        self.batches = batches

        self.procs = []
        self.done  = 0

        # be safe with concurrent iterators
        which = f'{which}_{id(self)}'

        self.q = Queue(1024)
        self.p = Process(target=_fill, args=(which, input_dim, self.q, batches))

        self.p.daemon = True # makes it easier to deal wih not actually cleaning up
        self.p.start()

        # FIXME need to drain the queue if we exit early to clean up shms

    def __next__(self):
        # we're done when we hit the _first_ done
        # because we always know which queue to target

        with self.tracer.start("queue_wait"):
            r = self.q.get()
            if r is None:
                self.p.join()
                self.p.close()
                return None

        i, shm_name, build, read = r
        assert i == self.nxt, f'expected {self.nxt} but got {i}'

        with self.tracer.start("shm_load"):
            batch_entries = self.batches[self.nxt] # to figure out the shape
            batch_size = len(batch_entries)

            shm = open_shm(name=shm_name)
            batch = np.ndarray((batch_size, self.input_dim), dtype=np.int64, buffer=shm)

        self.nxt += 1

        pid=id(self) % 32768
        self.tracer.record_manual("build_batch", build[0], build[1], pid=pid)

        if read[0] is not None:
            self.tracer.record_manual("load_file", read[0], read[1], pid=pid)

        with self.tracer.start("TF_convert"):
            ret = torch.Tensor(batch)

        shm.close()
        return ret

class EntryIteratorNoThreads(object):
    def __init__(self, input_dim, batches, tracer):
        self.input_dim     = input_dim
        self.batches       = iter(batches)
        self.tracer        = tracer
        self.open_filename = None
        self.data          = None

    def __next__(self):
        batch_entries = next(self.batches)
        batch_size    = len(batch_entries)
        batch         = np.zeros( (batch_size, self.input_dim) )

        with self.tracer.start("build_batch"):
            for entry_idx, entry in enumerate(batch_entries):
                file_name = entry[0]
                file_idx  = entry[1]

                if file_name != self.open_filename:
                    with self.tracer.start("load_file"):
                        with open(file_name, 'rb') as f:
                            decompressed_data = pyzstd.decompress(f.read())
                            buf = np.frombuffer(decompressed_data, dtype=np.float32)

                            self.open_filename = file_name
                            self.data = buf.reshape( (len(buf) // self.input_dim, self.input_dim) )

                batch[entry_idx,:] = self.data[file_idx,:]

        with self.tracer.start("TF_convert"):
            return torch.Tensor(batch)


class BoardDirLoader(object):
    """
    loads board data from the split up board files we have to produce.
    not enough ram to while entire dataset in memory for the large lookback models
    """

    def __init__(self, which, boards_dir, boards_per_file, input_dim, indices, batch_size, tracer, use_threads=True):
        """
        boards_dir: directory with boards files in it
        boards_per_file: how many entries make it into each boards file
        input_dim: dimensions of each entry
        indicies: indicies to fetch from the files, in the order they are requested
        """

        self.which       = which
        self.input_dim   = input_dim
        self.tracer      = tracer
        self.use_threads = use_threads

        with tracer.start("prepare_batches"):
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

        # prefetch
        if self.use_threads:
            self.nxt = EntryIterator(self.which, self.input_dim, self.batches, self.tracer)

    def __iter__(self):
        if self.use_threads:
            ret = self.nxt
            self.nxt = EntryIterator(self.which, self.input_dim, self.batches, self.tracer)
            return ret
        else:
            return EntryIteratorNoThreads(self.input_dim, self.batches, self.tracer)
