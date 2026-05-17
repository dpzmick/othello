import time
import os
import contextlib
import torch

class Tracer(object):
    def __init__(self, fname, enabled=False):
        self.enabled = enabled

        if self.enabled:
            self.f = open(fname, 'w')
            self.start_tm = time.time()
            self.empty = True
            self.f.write('[')

    @contextlib.contextmanager
    def start(self, event):
        if not self.enabled:
            yield
            return

        now = int((time.time() - self.start_tm) * 1000000)
        pid = os.getpid()

        if not self.empty:
            self.f.write(",")
        self.empty = False


        self.f.write(f'{{"name": "{event}", "ph": "B", "pid": {pid}, "tid": {pid}, "ts": {now}}}')

        yield

        now = int((time.time() - self.start_tm) * 1000000)
        self.f.write(f',{{"name": "{event}", "ph": "E", "pid": {pid}, "tid": {pid}, "ts": {now}}}')

    def record_manual(self, event, st, ed, pid=None):
        if not self.enabled:
            return

        pid = os.getpid() if pid is None else pid

        st = int( (st - self.start_tm) * 1000000)
        ed = int( (ed - self.start_tm) * 1000000)

        if st < 0 or ed < 0:
            raise RuntimeError(f"{st} or {ed} is negative")

        if not self.empty:
            self.f.write(",")
        self.empty = False

        self.f.write(f'{{"name": "{event}", "ph": "B", "pid": {pid}, "tid": {pid}, "ts": {st}}}')
        self.f.write(f',{{"name": "{event}", "ph": "E", "pid": {pid}, "tid": {pid}, "ts": {ed}}}')

    def close(self):
        if not self.enabled:
            return

        self.f.write(']')
        self.f.close()

if __name__ == "__main__":
    t = Tracer("out.trace")

    with t.start('a', 'c') as _e:
        with t.start('b', 'c'):
            print('asd')
        with t.start('b', 'c'):
            print('asd')
        with t.start('b2', 'c'):
            print('asd')
        with t.start('b', 'c'):
            print('asd')

    with t.start('a', 'c') as _e:
        with t.start('b', 'c'):
            print('asd')
        with t.start('b', 'c'):
            print('asd')
        with t.start('b2', 'c'):
            print('asd')
        with t.start('b', 'c'):
            print('asd')

    t.close()
