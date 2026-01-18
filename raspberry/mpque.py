from multiprocessing import Condition
import time


class MPDeque:
    def __init__(self, shared_list, cond, maxlen=None):
        self.data = shared_list
        self.cond = cond
        self.maxlen = maxlen

    def append(self, item):
        with self.cond:
            if self.maxlen is not None:
                while len(self.data) >= self.maxlen:
                    self.data.pop(0)
            self.data.append(item)
            self.cond.notify()

    def appendleft(self, item):
        with self.cond:
            if self.maxlen is not None:
                while len(self.data) >= self.maxlen:
                    self.data.pop()
            self.data.insert(0, item)
            self.cond.notify()

    def popright(self, block=True, timeout=None):
        return self._pop(left=False, block=block, timeout=timeout)

    def popleft(self, block=True, timeout=None):
        return self._pop(left=True, block=block, timeout=timeout)

    def _pop(self, left, block, timeout):
        with self.cond:
            if not block:
                if not self.data:
                    raise IndexError("empty")

            else:
                end = None if timeout is None else time.monotonic() + timeout
                while not self.data:
                    remaining = None if end is None else end - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError()
                    self.cond.wait(remaining)

            if left:
                return self.data.pop(0)
            else:
                return self.data.pop()
