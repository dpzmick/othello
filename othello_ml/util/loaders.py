class PrefetchDataLoaderIterator(object):
    def __init__(self, iterable):
        self.iter = iter(iterable)

        self.next = None
        self.prefetch() # start the load

    def prefetch(self):
        self.next = map(lambda e: e.to("cuda", non_blocking=True), self.iter.__next__())

    def __next__(self):
        ret = self.next
        self.prefetch()
        return ret

class PrefetchDataLoader(object):
    def __init__(self, dset, batch_size):
        iterable = []
        for i in range(0, int(len(dset)/batch_size)+1):
            st = i * batch_size
            ed = min((i+1) * batch_size, len(dset))

            if ed <= st:
                break

            x = dset[st:ed]

            if len(x[1]):
                iterable.append(x)

        self._inner = iterable

    def __iter__(self):
        return PrefetchDataLoaderIterator(self._inner)

    def __len__(self):
        return len(self._inner)

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

            if len(x[1]):
                iterable.append(x)

        self._inner = iterable

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)
